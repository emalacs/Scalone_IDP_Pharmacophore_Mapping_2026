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

// Pharmacophore (contested space) is the more commonly inspected of the two --
// default to it rather than the SPACES array's own (Growth space first) order,
// so a reorder of that array doesn't silently change the default.
const DEFAULT_SPACE = SPACES.find((s) => s.id === 'contested') ?? SPACES[0]

export default function App() {
  const [subsetId, setSubsetId] = useState(SUBSETS[0].id)
  const [spaceId, setSpaceId] = useState(DEFAULT_SPACE.id)
  const [variantId, setVariantId] = useState(DEFAULT_SPACE.variants[0].id)
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
            src={`${import.meta.env.BASE_URL}dartmouth-logo.svg`}
            alt="Dartmouth College"
            className="h-9 w-auto"
            onError={(e) => {
              e.currentTarget.style.display = 'none'
            }}
          />
          <div className="text-xs uppercase tracking-wide text-white/50">Department of Chemistry · Dartmouth College</div>
          <div className="space-y-1.5">
            <p className="text-sm font-medium leading-snug text-white">
              Dynamic Pharmacophore Mapping for Intrinsically Disordered Drug Targets
            </p>
            <div className="space-y-1 text-xs">
              {/* TODO: paper URL, still to be provided */}
              <a href="#" className="block text-dartmouth-green underline decoration-dartmouth-green/50 hover:text-white">
                Paper
              </a>
              <div className="text-white/50">Supplementary Information</div>
              <ul className="space-y-0.5 pl-3">
                <li>
                  <a
                    href="https://github.com/emalacs/Scalone_IDP_Pharmacophore_Mapping_2026"
                    className="text-dartmouth-green underline decoration-dartmouth-green/50 hover:text-white"
                  >
                    Repository
                  </a>
                </li>
                <li>
                  <a
                    href="https://colab.research.google.com/github/emalacs/Scalone_IDP_Pharmacophore_Mapping_2026/blob/main/notebooks/01_split_trajectory_pca_graph.ipynb"
                    className="text-dartmouth-green underline decoration-dartmouth-green/50 hover:text-white"
                  >
                    Colab · Split trajectory (PCA + Graph)
                  </a>
                </li>
                <li>
                  <a
                    href="https://colab.research.google.com/github/emalacs/Scalone_IDP_Pharmacophore_Mapping_2026/blob/main/notebooks/02_pharmacophore_from_subset.ipynb"
                    className="text-dartmouth-green underline decoration-dartmouth-green/50 hover:text-white"
                  >
                    Colab · Pharmacophore from subset
                  </a>
                </li>
                {/* TODO: link once the Zenodo deposit + DOI exist */}
                <li className="text-white/40">Zenodo (DOI pending)</li>
              </ul>
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

          <div>
            <span className={labelClass}>Pharmacophore space</span>
            <div className="mt-1 grid grid-cols-2 gap-1 rounded border border-white/20 bg-white/5 p-1 text-sm">
              {SPACES.map((s) => (
                <button
                  key={s.id}
                  type="button"
                  onClick={() => handleSpaceChange(s.id)}
                  aria-pressed={spaceId === s.id}
                  className={`rounded px-2 py-1.5 transition-colors ${
                    spaceId === s.id ? 'bg-dartmouth-green text-white' : 'text-white/60 hover:text-white'
                  }`}
                >
                  {s.label}
                </button>
              ))}
            </div>
            <p className="mt-1 text-xs text-white/40">{SPACE_DEFINITIONS}</p>
          </div>

          {/* Negative-space layer toggles + contour sliders, rendered by MolstarViewer via portal
              once each layer's volume stats are known. */}
          <div className="space-y-2">
            <span className={labelClass}>Show</span>
            <div ref={setLayerSlotNode} className="space-y-3" />
          </div>

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
