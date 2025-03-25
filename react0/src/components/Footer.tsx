import React from 'react'

interface FooterProps {
  appVersion?: string
  envName?: string
}

const Footer: React.FC<FooterProps> = ({ appVersion, envName }) => {
  const year = new Date().getFullYear()

  return (
    <footer style={{
      textAlign: 'center',
      padding: '15px 20px',
      backgroundColor: '#f0f0f0',
      color: '#333',
      marginTop: 'auto'
    }}>
      <p style={{ fontSize: '0.9rem', margin: 0 }}>
        &copy; {year} MyDiabetesProject. All rights reserved.
      </p>
      {appVersion || envName ? (
        <p style={{ fontSize: '0.8rem', margin: '4px 0 0' }}>
          {appVersion && `Version: ${appVersion}`}
          {envName && ` ${appVersion ? '|' : ''} Environment: ${envName}`}
        </p>
      ) : null}
    </footer>
  )
}

export default Footer
