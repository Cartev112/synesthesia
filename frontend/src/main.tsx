import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import './index.css'
import { AudioEngineProvider } from './contexts/AudioEngineContext'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <AudioEngineProvider>
      <App />
    </AudioEngineProvider>
  </React.StrictMode>,
)
