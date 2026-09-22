import { useMemo, useState } from 'react'
import MolstarViewer from './components/MolstarViewer'
import {
  FEATURES,
  NEGATIVE_SPACE_LAYERS,
  RESOLUTIONS,
  SPACE_DEFINITIONS,
  SPACES,
  SUBSETS,
  featureUrl,
  ligandUrl,
  negativeSpaceUrl,
} from './data/catalog'

const labelClass = 'block text-xs font-medium uppercase tracking-wide text-white/50'
const selectClass =
  'mt-1 w-full rounded border border-white/20 bg-white/5 px-2 py-1.5 text-sm text-white focus:border-dartmouth-green focus:outline-none focus:ring-1 focus:ring-dartmouth-green'

export default function App() {
  const [subsetId, setSubsetId] = useState(SUBSETS[0].id)
  const [spaceId, setSpaceId] = useState(SPACES[0].id)
  const [variantId, setVariantId] = useState(SPACES[0].variants[0].id)
  const [featureId, setFeatureId] = useState(FEATURES[0].id)
  const [resolutionId, setResolutionId] = useState(RESOLUTIONS[3].id) // top5pct
  const [visibleLayerIds, setVisibleLayerIds] = useState(() => new Set())
  const [layerSlotNode, setLayerSlotNode] = useState(null)

  const subset = SUBSETS.find((s) => s.id === subsetId)
  const space = SPACES.find((s) => s.id === spaceId)
  const variant = space.variants.find((v) => v.id === variantId) ?? space.variants[0]
  const feature = FEATURES.find((f) => f.id === featureId)
  const resolution = RESOLUTIONS.find((r) => r.id === resolutionId)

  // Each space has its own set of variants; switching space resets to that
  // space's default (plain) rather than carrying over a variant id that may
  // not exist there.
  const handleSpaceChange = (id) => {
    setSpaceId(id)
    const newSpace = SPACES.find((s) => s.id === id)
    setVariantId(newSpace.variants[0].id)
  }

  // A non-plain variant (contacts, ratio_*, diff) has its own established
  // color, overriding the feature's own color for that selection. Memoized so
  // MolstarViewer's reload effect (keyed on this object's identity) doesn't
  // fire on every render -- e.g. a layer-checkbox toggle -- only when the
  // feature or its effective color actually changes.
  const displayColor = variant.color ?? feature.color
  const displayFeature = useMemo(() => ({ ...feature, color: displayColor }), [feature, displayColor])

  const layers = useMemo(
    () =>
      NEGATIVE_SPACE_LAYERS.map((layer) => ({
        id: layer.id,
        label: layer.label,
        color: layer.color,
        url: negativeSpaceUrl(subset, layer),
        visible: visibleLayerIds.has(layer.id),
      })),
    [subset, visibleLayerIds],
  )

  const toggleLayer = (id) => {
    setVisibleLayerIds((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  return (
    <div className="flex min-h-screen bg-white text-neutral-900">
      <aside className="flex w-80 flex-shrink-0 flex-col gap-6 overflow-y-auto bg-rich-forest px-5 py-6 text-white">
        {/* Institution header */}
        <div className="space-y-3">
          <img
            src="/dartmouth-logo.svg"
            alt="Dartmouth College"
            className="h-9 w-auto"
            onError={(e) => {
              e.currentTarget.style.display = 'none'
            }}
          />
          {/* TODO: replace with the actual lab/department affiliation */}
          <div className="text-xs uppercase tracking-wide text-white/50">Department of Chemistry · Dartmouth College</div>
          <div className="space-y-1.5">
            <p className="text-sm font-medium leading-snug text-white">
              Dynamic Pharmacophore Mapping for Intrinsically Disordered Drug Targets
            </p>
            {/* TODO: paper + SI links, to be provided */}
            <div className="flex gap-3 text-xs">
              <a href="#" className="text-dartmouth-green underline decoration-dartmouth-green/50 hover:text-white">
                Paper
              </a>
              <a href="#" className="text-dartmouth-green underline decoration-dartmouth-green/50 hover:text-white">
                Supplementary Information
              </a>
            </div>
          </div>
        </div>

        <hr className="border-white/10" />

        {/* Settings */}
        <div className="space-y-4">
          <div className="text-xs uppercase tracking-wide text-white/50">Pharmacophore Viewer</div>

          <label className="block">
            <span className={labelClass}>Subset</span>
            <select className={selectClass} value={subsetId} onChange={(e) => setSubsetId(e.target.value)}>
              {SUBSETS.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.label}
                </option>
              ))}
            </select>
          </label>

          <label className="block">
            <span className={labelClass}>Pharmacophore space</span>
            <select className={selectClass} value={spaceId} onChange={(e) => handleSpaceChange(e.target.value)}>
              {SPACES.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.label}
                </option>
              ))}
            </select>
            <p className="mt-1 text-xs text-white/40">{SPACE_DEFINITIONS}</p>
          </label>

          {space.variants.length > 1 && (
            <label className="block">
              <span className={labelClass}>Variant</span>
              <select className={selectClass} value={variantId} onChange={(e) => setVariantId(e.target.value)}>
                {space.variants.map((v) => (
                  <option key={v.id} value={v.id}>
                    {v.label}
                  </option>
                ))}
              </select>
            </label>
          )}

          <label className="block">
            <span className={labelClass}>Feature map</span>
            <select className={selectClass} value={featureId} onChange={(e) => setFeatureId(e.target.value)}>
              {FEATURES.map((f) => (
                <option key={f.id} value={f.id}>
                  {f.label}
                </option>
              ))}
            </select>
          </label>

          <label className="block">
            <span className={labelClass}>Resolution</span>
            <select className={selectClass} value={resolutionId} onChange={(e) => setResolutionId(e.target.value)}>
              {RESOLUTIONS.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.label}
                </option>
              ))}
            </select>
          </label>
        </div>

        <hr className="border-white/10" />

        {/* Negative-space layer toggles + contour sliders, rendered by MolstarViewer via portal
            once each layer's volume stats are known. */}
        <div className="space-y-3">
          <div className={labelClass}>Negative space</div>
          <div ref={setLayerSlotNode} className="space-y-3" />
        </div>
      </aside>

      <main className="flex-1 p-6">
        <div className="relative h-[calc(100vh-3rem)] w-full overflow-hidden rounded border border-neutral-200">
          <MolstarViewer
            key={subsetId}
            ligandUrl={ligandUrl(subset)}
            featureUrl={featureUrl(subset, space, feature, variant, resolution)}
            feature={displayFeature}
            layers={layers}
            onToggleLayer={toggleLayer}
            layerControlsSlot={layerSlotNode}
          />
        </div>
      </main>
    </div>
  )
}
