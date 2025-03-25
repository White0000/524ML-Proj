import React, { useState } from 'react'
import { Link } from 'react-router-dom'

interface NavLink {
  label: string
  path: string
}

interface HeaderProps {
  brandName?: string
  brandLogo?: string
  links?: NavLink[]
}

const Header: React.FC<HeaderProps> = ({
  brandName = "MyDiabetesProject",
  brandLogo,
  links = [
    { label: "Home", path: "/" },
    { label: "About", path: "/about" },
    { label: "Dashboard", path: "/dashboard" },
    { label: "Monitor", path: "/monitor" }
  ]
}) => {
  const [mobileOpen, setMobileOpen] = useState(false)

  return (
    <header style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      backgroundColor: '#007bff',
      color: '#fff',
      padding: '10px 20px',
      position: 'relative'
    }}>
      <div style={{ display: 'flex', alignItems: 'center' }}>
        {brandLogo && (
          <img
            src={brandLogo}
            alt="logo"
            style={{ height: '40px', marginRight: '10px' }}
          />
        )}
        <div style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>
          {brandName}
        </div>
      </div>
      <nav style={{ display: 'none', gap: '15px' }} className="desktop-nav">
        {links.map((link) => (
          <Link
            key={link.label}
            to={link.path}
            style={{ color: '#fff', textDecoration: 'none' }}
          >
            {link.label}
          </Link>
        ))}
      </nav>
      <button
        style={{
          display: 'none',
          background: 'none',
          border: 'none',
          color: '#fff',
          fontSize: '1.2rem'
        }}
        className="mobile-toggle"
        onClick={() => setMobileOpen(!mobileOpen)}
      >
        ☰
      </button>
      <div
        style={{
          position: 'absolute',
          top: '60px',
          right: 0,
          backgroundColor: '#007bff',
          display: mobileOpen ? 'flex' : 'none',
          flexDirection: 'column',
          alignItems: 'flex-start',
          padding: '10px'
        }}
        className="mobile-nav"
      >
        {links.map((link) => (
          <Link
            key={link.label}
            to={link.path}
            style={{
              color: '#fff',
              textDecoration: 'none',
              margin: '5px 0'
            }}
            onClick={() => setMobileOpen(false)}
          >
            {link.label}
          </Link>
        ))}
      </div>
      <style>
        {`
          @media (min-width: 768px) {
            .desktop-nav {
              display: flex !important;
            }
            .mobile-toggle {
              display: none !important;
            }
          }
          @media (max-width: 767px) {
            .mobile-toggle {
              display: block !important;
            }
          }
        `}
      </style>
    </header>
  )
}

export default Header
