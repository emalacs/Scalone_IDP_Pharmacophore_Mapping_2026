import { ungzip } from 'pako'
import { Volume } from 'molstar/lib/mol-model/volume'
import { createVolumeRepresentationParams } from 'molstar/lib/mol-plugin-state/helpers/volume-representation-params'
import { StateTransforms } from 'molstar/lib/mol-plugin-state/transforms'
import { Color } from 'molstar/lib/mol-util/color'

export async function loadLigandStructure(plugin, url) {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status}`)
  const pdbText = await res.text()

  const data = await plugin.builders.data.rawData({ data: pdbText, label: 'Ligand' })
  const trajectory = await plugin.builders.structure.parseTrajectory(data, 'pdb')
  const preset = await plugin.builders.structure.hierarchy.applyPreset(trajectory, 'default', {
    structure: { name: 'model', params: {} },
    showUnitcell: false,
    representationPreset: 'auto',
  })
  return preset
}

// Default contour level for a freshly loaded volume: mean + 1 sigma, clamped
// into the volume's own value range. Adapts automatically across full-res vs.
// thinned files and across feature types, whose ranges differ by 10x or more.
export function defaultIsoValue(stats) {
  const value = stats.mean + stats.sigma
  return Math.min(stats.max, Math.max(stats.min, value))
}

export async function loadVolumeIsosurface(plugin, url, { isoValue, color, alpha = 0.45 } = {}) {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status}`)
  const raw = new Uint8Array(await res.arrayBuffer())
  // Some static servers (Vite's dev/preview server included) auto-decompress
  // .gz files and set Content-Encoding: gzip, which makes fetch() hand back
  // already-decompressed bytes. Only ungzip if the gzip magic header is present.
  const isGzipped = raw[0] === 0x1f && raw[1] === 0x8b
  const bytes = isGzipped ? ungzip(raw) : raw

  const binaryData = await plugin.builders.data.rawData({ data: bytes, label: 'Volume' })
  const format = plugin.build().to(binaryData).apply(StateTransforms.Data.ParseCcp4, {}, { state: { isGhost: true } })
  const volume = format.apply(StateTransforms.Volume.VolumeFromCcp4, {})
  await format.commit({ revertOnError: true })

  const stats = volume.selector.data.grid.stats
  const resolvedIsoValue = isoValue ?? defaultIsoValue(stats)

  const repr = plugin.build().to(volume).apply(
    StateTransforms.Representation.VolumeRepresentation3D,
    createVolumeRepresentationParams(plugin, volume.selector.data, {
      type: 'isosurface',
      typeParams: { isoValue: Volume.IsoValue.absolute(resolvedIsoValue), alpha },
      color: 'uniform',
      colorParams: { value: Color(color) },
    }),
  )
  // repr is a StateBuilder.To, consumed by commit() below. Keep .selector (a
  // stable StateObjectSelector that mints a fresh builder per call) for later
  // live updates — reusing `repr` itself would hit "already applied" errors.
  const reprSelector = repr.selector
  await repr.commit()

  return { binaryData, format, volume, repr: reprSelector, stats, isoValue: resolvedIsoValue }
}

export async function updateIsosurfaceLevel(reprSelector, isoValue) {
  // The committed transform params are shaped { type: { name, params }, colorTheme, sizeTheme }
  // (see createVolumeRepresentationParams) — isoValue lives at type.params.isoValue,
  // not at a top-level typeParams key.
  await reprSelector
    .update((old) => {
      old.type.params.isoValue = Volume.IsoValue.absolute(isoValue)
    })
    .commit()
}

export async function unloadVolume(plugin, binaryData) {
  await plugin.build().delete(binaryData).commit()
}

export async function clearPlugin(plugin) {
  await plugin.clear()
}
