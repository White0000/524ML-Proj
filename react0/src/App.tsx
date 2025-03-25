import React from 'react'
import { Provider } from 'react-redux'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { store } from './store'
import Home from './pages/Home'
import About from './pages/About'
import Dashboard from './pages/Dashboard'
import MetricsMonitor from './pages/MetricsMonitor'
import NotFound from './pages/NotFound'
import Header from './components/Header'
import Sidebar from './components/Sidebar'
import Footer from './components/Footer'

const App: React.FC = () => {
  return (
    <Provider store={store}>
      <Router>
        <div style={{ display: 'flex', minHeight: '100vh', width: '100%' }}>
          <Sidebar />
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
            <Header />
            <div style={{ flex: 1, padding: '20px' }}>
              <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/about" element={<About />} />
                <Route path="/dashboard" element={<Dashboard />} />
                <Route path="/monitor" element={<MetricsMonitor />} />
                <Route path="*" element={<NotFound />} />
              </Routes>
            </div>
            <Footer />
          </div>
        </div>
      </Router>
    </Provider>
  )
}

export default App
