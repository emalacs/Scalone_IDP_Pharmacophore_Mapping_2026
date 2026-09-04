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

export default function App() {
  const [subsetId, setSubsetId] = useState(SUBSETS[0].id)
  const [spaceId, setSpaceId] = useState(SPACES[0].id)
  const [variantId, setVariantId] = useState(SPACES[0].variants[0].id)
  const [featureId, setFeatureId] = useState(FEATURES[0].id)
  const [resolutionId, setResolutionId] = useState(RESOLUTIONS[3].id) // top5pct
  const [visibleLayerIds, setVisibleLayerIds] = useState(() => new Set())

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
    <div className="min-h-screen bg-neutral-950 text-neutral-100">
      <div className="mx-auto max-w-5xl px-4 py-6">
        <h1 className="text-lg font-semibold tracking-wide text-neutral-200">
          Pharmacophore Viewer <span className="text-neutral-500">· proof of concept</span>
        </h1>
        <p className="mt-1 text-sm text-neutral-500">{SPACE_DEFINITIONS}</p>

        <div className="mt-4 flex flex-wrap items-center gap-4">
          <label className="flex items-center gap-2 text-sm text-neutral-400">
            Subset
            <select
              className="rounded border border-neutral-700 bg-neutral-900 px-2 py-1 text-neutral-100"
              value={subsetId}
              onChange={(e) => setSubsetId(e.target.value)}
            >
              {SUBSETS.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.label}
                </option>
              ))}
            </select>
          </label>

          <label className="flex items-center gap-2 text-sm text-neutral-400">
            Pharmacophore space
            <select
              className="rounded border border-neutral-700 bg-neutral-900 px-2 py-1 text-neutral-100"
              value={spaceId}
              onChange={(e) => handleSpaceChange(e.target.value)}
            >
              {SPACES.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.label}
                </option>
              ))}
            </select>
          </label>

          {space.variants.length > 1 && (
            <label className="flex items-center gap-2 text-sm text-neutral-400">
              Variant
              <select
                className="rounded border border-neutral-700 bg-neutral-900 px-2 py-1 text-neutral-100"
                value={variantId}
                onChange={(e) => setVariantId(e.target.value)}
              >
                {space.variants.map((v) => (
                  <option key={v.id} value={v.id}>
                    {v.label}
                  </option>
                ))}
              </select>
            </label>
          )}

          <label className="flex items-center gap-2 text-sm text-neutral-400">
            Feature map
            <select
              className="rounded border border-neutral-700 bg-neutral-900 px-2 py-1 text-neutral-100"
              value={featureId}
              onChange={(e) => setFeatureId(e.target.value)}
            >
              {FEATURES.map((f) => (
                <option key={f.id} value={f.id}>
                  {f.label}
                </option>
              ))}
            </select>
          </label>

          <label className="flex items-center gap-2 text-sm text-neutral-400">
            Resolution
            <select
              className="rounded border border-neutral-700 bg-neutral-900 px-2 py-1 text-neutral-100"
              value={resolutionId}
              onChange={(e) => setResolutionId(e.target.value)}
            >
              {RESOLUTIONS.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.label}
                </option>
              ))}
            </select>
          </label>

          <div className="flex items-center gap-3 text-sm text-neutral-400">
            {NEGATIVE_SPACE_LAYERS.map((layer) => (
              <label key={layer.id} className="flex items-center gap-1.5">
                <input
                  type="checkbox"
                  checked={visibleLayerIds.has(layer.id)}
                  onChange={() => toggleLayer(layer.id)}
                />
                <span
                  className="inline-block h-2.5 w-2.5 rounded-full"
                  style={{ backgroundColor: `#${layer.color.toString(16).padStart(6, '0')}` }}
                />
                {layer.label}
              </label>
            ))}
          </div>
        </div>

        <div className="relative mt-4 h-[65vh] min-h-[420px] w-full overflow-hidden rounded border border-neutral-800">
          <MolstarViewer
            key={subsetId}
            ligandUrl={ligandUrl(subset)}
            featureUrl={featureUrl(subset, space, feature, variant, resolution)}
            feature={displayFeature}
            layers={layers}
          />
        </div>
      </div>
    </div>
  )
}
