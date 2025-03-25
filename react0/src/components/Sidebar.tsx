import React, { useState } from 'react'
import { Link, useLocation } from 'react-router-dom'

interface SidebarLink {
  label: string
  path: string
  icon?: React.ReactNode
}

interface SidebarProps {
  links?: SidebarLink[]
  width?: string
  collapsible?: boolean
  defaultCollapsed?: boolean
  backgroundColor?: string
  textColor?: string
  highlightColor?: string
}

const Sidebar: React.FC<SidebarProps> = ({
  links = [
    { label: 'Home', path: '/' },
    { label: 'About', path: '/about' },
    { label: 'Dashboard', path: '/dashboard' },
    { label: 'Monitor', path: '/monitor' }
  ],
  width = '200px',
  collapsible = true,
  defaultCollapsed = false,
  backgroundColor = '#f7f7f7',
  textColor = '#333',
  highlightColor = '#007bff'
}) => {
  const [collapsed, setCollapsed] = useState(defaultCollapsed)
  const location = useLocation()

  const toggleCollapse = () => {
    if (!collapsible) return
    setCollapsed(!collapsed)
  }

  return (
    <aside
      style={{
        width: collapsed ? '60px' : width,
        backgroundColor,
        borderRight: '1px solid #ccc',
        height: '100vh',
        boxSizing: 'border-box',
        display: 'flex',
        flexDirection: 'column',
        transition: 'width 0.2s ease-in-out',
        position: 'relative'
      }}
    >
      {collapsible && (
        <button
          onClick={toggleCollapse}
          style={{
            position: 'absolute',
            top: '10px',
            right: '-15px',
            width: '30px',
            height: '30px',
            backgroundColor: highlightColor,
            color: '#fff',
            border: 'none',
            borderRadius: '50%',
            cursor: 'pointer',
            fontSize: '1rem'
          }}
        >
          {collapsed ? '>' : '<'}
        </button>
      )}
      <nav style={{
        marginTop: '60px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: collapsed ? 'center' : 'flex-start',
        padding: '20px'
      }}>
        {links.map(({ label, path, icon }) => {
          const isActive = location.pathname === path
          return (
            <Link
              key={label}
              to={path}
              style={{
                marginBottom: '10px',
                textDecoration: 'none',
                color: isActive ? highlightColor : textColor,
                fontWeight: isActive ? 'bold' : 'normal',
                width: '100%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: collapsed ? 'center' : 'flex-start',
                transition: 'color 0.2s'
              }}
            >
              {icon && <span style={{ marginRight: collapsed ? 0 : '8px' }}>{icon}</span>}
              {!collapsed && label}
            </Link>
          )
        })}
      </nav>
    </aside>
  )
}

export default Sidebar
