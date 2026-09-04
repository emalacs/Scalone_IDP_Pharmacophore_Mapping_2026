import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

// No StrictMode: Mol*'s createPluginUI() calls createRoot() on our container
// imperatively and isn't safe to invoke twice on the same node, which
// StrictMode's intentional double-mount in dev would trigger.
createRoot(document.getElementById('root')).render(<App />)
