import React from 'react'
import { Link } from 'react-router-dom'

interface NotFoundProps {
  title?: string
  message?: string
  suggestions?: { label: string; path: string }[]
}

const NotFound: React.FC<NotFoundProps> = ({
  title = "404 - Not Found",
  message = "The page you are looking for doesn't exist or might have been moved.",
  suggestions = [
    { label: "Home", path: "/" },
    { label: "About", path: "/about" },
    { label: "Dashboard", path: "/dashboard" }
  ]
}) => {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        height: '80vh',
        fontFamily: 'Arial, sans-serif',
        textAlign: 'center',
        padding: '20px'
      }}
    >
      <h1 style={{ fontSize: '2rem', marginBottom: '1rem' }}>{title}</h1>
      <p style={{ fontSize: '1rem', maxWidth: '400px', margin: '0 auto 1.5rem' }}>
        {message}
      </p>
      <div>
        {suggestions.map(({ label, path }) => (
          <Link
            key={label}
            to={path}
            style={{
              display: 'inline-block',
              margin: '0 10px',
              padding: '8px 16px',
              borderRadius: '4px',
              backgroundColor: '#007bff',
              color: '#fff',
              textDecoration: 'none',
              transition: 'background-color 0.3s'
            }}
          >
            {label}
          </Link>
        ))}
      </div>
    </div>
  )
}

export default NotFound
