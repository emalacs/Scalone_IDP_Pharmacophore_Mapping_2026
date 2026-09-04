import { DefaultPluginUISpec } from 'molstar/lib/mol-plugin-ui/spec'
import { createPluginUI } from 'molstar/lib/mol-plugin-ui'
import { renderReact18 } from 'molstar/lib/mol-plugin-ui/react18'
import { PluginConfig } from 'molstar/lib/mol-plugin/config'
import { useEffect, useRef, useState } from 'react'
import {
  clearPlugin,
  loadLigandStructure,
  loadVolumeIsosurface,
  unloadVolume,
  updateIsosurfaceLevel,
} from '../lib/molstarLoaders'

const CONTOUR_UPDATE_DEBOUNCE_MS = 60

export default function MolstarViewer({ ligandUrl, feature, featureUrl, layers = [] }) {
  const containerRef = useRef(null)
  const pluginRef = useRef(null)
  const reprRef = useRef(null)
  const debounceTimer = useRef(null)
  const layerStateRef = useRef(new Map()) // layer id -> { binaryData, repr }
  const layersRef = useRef(layers)

  const [pluginReady, setPluginReady] = useState(false)
  const [loadState, setLoadState] = useState('loading')
  const [volumeStats, setVolumeStats] = useState(null)
  const [contour, setContour] = useState(0)

  useEffect(() => {
    layersRef.current = layers
  }, [layers])

  useEffect(() => {
    let disposed = false

    createPluginUI({
      target: containerRef.current,
      render: renderReact18,
      spec: {
        ...DefaultPluginUISpec(),
        config: [[PluginConfig.Viewport.ShowAnimation, false]],
      },
    }).then((plugin) => {
      if (disposed) {
        plugin.dispose()
        return
      }
      pluginRef.current = plugin
      setPluginReady(true)
    })

    return () => {
      disposed = true
      pluginRef.current?.dispose()
      pluginRef.current = null
    }
  }, [])

  // Primary reload: ligand + selected feature map, plus any layers already
  // toggled on. Runs whenever the subset/feature/resolution selection changes.
  useEffect(() => {
    const plugin = pluginRef.current
    if (!pluginReady || !plugin) return

    let cancelled = false
    setLoadState('loading')
    setVolumeStats(null)
    reprRef.current = null

    ;(async () => {
      try {
        await clearPlugin(plugin)
        layerStateRef.current.clear()
        if (cancelled) return

        await loadLigandStructure(plugin, ligandUrl)
        if (cancelled) return

        const { repr, stats, isoValue } = await loadVolumeIsosurface(plugin, featureUrl, {
          color: feature.color,
        })
        if (cancelled) return
        reprRef.current = repr
        setVolumeStats(stats)
        setContour(isoValue)

        for (const layer of layersRef.current) {
          if (!layer.visible) continue
          const { binaryData, repr: layerRepr } = await loadVolumeIsosurface(plugin, layer.url, {
            color: layer.color,
          })
          if (cancelled) return
          layerStateRef.current.set(layer.id, { binaryData, repr: layerRepr })
        }

        plugin.managers.camera.reset()
        setLoadState('ready')
      } catch (err) {
        // A superseded reload (e.g. rapid dropdown changes) can crash mid-flight
        // when a newer effect's clearPlugin() wipes state out from under this
        // one's in-progress fetch/parse -- expected, not worth logging.
        if (!cancelled) {
          console.error(err)
          setLoadState('error')
        }
      }
    })()

    return () => {
      cancelled = true
    }
  }, [pluginReady, ligandUrl, featureUrl, feature])

  // Layer toggles: add/remove individual extra volumes without touching the
  // ligand or the primary feature isosurface.
  useEffect(() => {
    const plugin = pluginRef.current
    if (!pluginReady || !plugin || loadState !== 'ready') return

    let cancelled = false

    ;(async () => {
      for (const layer of layers) {
        const loaded = layerStateRef.current.get(layer.id)
        if (layer.visible && !loaded) {
          try {
            const { binaryData, repr } = await loadVolumeIsosurface(plugin, layer.url, { color: layer.color })
            if (cancelled) return
            layerStateRef.current.set(layer.id, { binaryData, repr })
          } catch (err) {
            console.error(err)
          }
        } else if (!layer.visible && loaded) {
          layerStateRef.current.delete(layer.id)
          await unloadVolume(plugin, loaded.binaryData).catch(console.error)
        }
      }
    })()

    return () => {
      cancelled = true
    }
  }, [layers, pluginReady, loadState])

  const handleContourChange = (value) => {
    setContour(value)
    if (!reprRef.current) return

    clearTimeout(debounceTimer.current)
    debounceTimer.current = setTimeout(() => {
      updateIsosurfaceLevel(reprRef.current, value).catch(console.error)
    }, CONTOUR_UPDATE_DEBOUNCE_MS)
  }

  return (
    <div className="relative h-full w-full">
      <div ref={containerRef} className="h-full w-full" />
      {loadState === 'loading' && (
        <div className="pointer-events-none absolute top-2 left-2 rounded bg-black/70 px-2 py-1 text-xs text-white">
          Loading…
        </div>
      )}
      {loadState === 'error' && (
        <div className="pointer-events-none absolute top-2 left-2 rounded bg-red-700/80 px-2 py-1 text-xs text-white">
          Failed to load — see console
        </div>
      )}
      {volumeStats && (
        <div className="absolute bottom-2 left-2 flex items-center gap-2 rounded bg-black/70 px-3 py-2 text-xs text-white">
          <span className="whitespace-nowrap">Contour</span>
          <input
            type="range"
            min={volumeStats.min}
            max={volumeStats.max}
            step={(volumeStats.max - volumeStats.min) / 500 || 0.0001}
            value={contour}
            onChange={(e) => handleContourChange(Number(e.target.value))}
            className="w-40"
          />
          <span className="w-16 whitespace-nowrap tabular-nums">{contour.toPrecision(3)}</span>
        </div>
      )}
    </div>
  )
}
