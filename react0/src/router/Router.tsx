import React from 'react'
import { Routes, Route } from 'react-router-dom'
import Home from '../pages/Home'
import About from '../pages/About'
import NotFound from '../pages/NotFound'
import Dashboard from '../pages/Dashboard'
import MetricsMonitor from '../pages/MetricsMonitor'

const RouterConfig: React.FC = () => {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/about" element={<About />} />
      <Route path="/dashboard" element={<Dashboard />} />
      <Route path="/monitor" element={<MetricsMonitor />} />
      <Route path="*" element={<NotFound />} />
    </Routes>
  )
}

export default RouterConfig
