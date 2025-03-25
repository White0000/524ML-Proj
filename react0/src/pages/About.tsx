import React from 'react'

interface TeamMember {
  name: string
  role: string
  avatarUrl?: string
}

const teamMembers: TeamMember[] = [
  { name: 'Alice', role: 'Data Engineer', avatarUrl: '' },
  { name: 'Bob', role: 'ML Engineer', avatarUrl: '' },
  { name: 'Carol', role: 'Frontend Developer', avatarUrl: '' },
  { name: 'Dave', role: 'DevOps Engineer', avatarUrl: '' },
]

const About: React.FC = () => {
  return (
    <div style={{
      padding: '30px',
      fontFamily: 'Arial, sans-serif',
      maxWidth: '900px',
      margin: '0 auto'
    }}>
      <section style={{ textAlign: 'center', marginBottom: '2rem' }}>
        <h2 style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>About This Project</h2>
        <p style={{ fontSize: '1rem', lineHeight: 1.6 }}>
          This application is designed to help detect the risk of diabetes using predictive models.
          It integrates a Python backend for data processing, training, and inference, with a modern
          React frontend for an intuitive user experience. The project aims to bridge medical insights
          with practical software solutions, enhancing early detection and health management.
        </p>
      </section>

      <section style={{ marginBottom: '2rem' }}>
        <h3 style={{ fontSize: '1.5rem', marginBottom: '1rem' }}>Our Goals</h3>
        <ul style={{ fontSize: '1rem', lineHeight: 1.6, listStyleType: 'disc', paddingLeft: '20px' }}>
          <li>Accurately predict the risk of diabetes through robust machine learning models</li>
          <li>Provide a user-friendly interface for healthcare professionals and patients</li>
          <li>Enable data-driven insights to inform clinical decisions</li>
          <li>Maintain data privacy and adhere to medical regulations</li>
        </ul>
      </section>

      <section style={{ marginBottom: '2rem' }}>
        <h3 style={{ fontSize: '1.5rem', marginBottom: '1rem' }}>Technology Stack</h3>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem' }}>
          <div style={{ flex: '1 1 300px', backgroundColor: '#f5f5f5', padding: '10px', borderRadius: '4px' }}>
            <strong>Backend:</strong>
            <p style={{ margin: '0.5rem 0' }}>Python, Flask/FastAPI, scikit-learn, XGBoost</p>
          </div>
          <div style={{ flex: '1 1 300px', backgroundColor: '#f5f5f5', padding: '10px', borderRadius: '4px' }}>
            <strong>Frontend:</strong>
            <p style={{ margin: '0.5rem 0' }}>React, TypeScript, Chart.js, CSS-in-JS</p>
          </div>
          <div style={{ flex: '1 1 300px', backgroundColor: '#f5f5f5', padding: '10px', borderRadius: '4px' }}>
            <strong>Data & Storage:</strong>
            <p style={{ margin: '0.5rem 0' }}>MySQL/PostgreSQL, Docker, S3 (optional)</p>
          </div>
          <div style={{ flex: '1 1 300px', backgroundColor: '#f5f5f5', padding: '10px', borderRadius: '4px' }}>
            <strong>DevOps:</strong>
            <p style={{ margin: '0.5rem 0' }}>Docker, CI/CD pipelines, Nginx</p>
          </div>
        </div>
      </section>

      <section style={{ marginBottom: '2rem' }}>
        <h3 style={{ fontSize: '1.5rem', marginBottom: '1rem' }}>Meet the Team</h3>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem' }}>
          {teamMembers.map((member) => (
            <div
              key={member.name}
              style={{
                flex: '1 1 200px',
                backgroundColor: '#fafafa',
                padding: '10px',
                borderRadius: '4px',
                textAlign: 'center'
              }}
            >
              {member.avatarUrl ? (
                <img
                  src={member.avatarUrl}
                  alt={member.name}
                  style={{ width: '80px', height: '80px', borderRadius: '50%' }}
                />
              ) : (
                <div
                  style={{
                    width: '80px',
                    height: '80px',
                    borderRadius: '50%',
                    backgroundColor: '#ccc',
                    margin: '0 auto'
                  }}
                />
              )}
              <p style={{ margin: '0.5rem 0', fontWeight: 'bold' }}>{member.name}</p>
              <p style={{ margin: '0.5rem 0', fontSize: '0.9rem', color: '#666' }}>{member.role}</p>
            </div>
          ))}
        </div>
      </section>

      <section>
        <h3 style={{ fontSize: '1.5rem', marginBottom: '1rem' }}>Next Steps</h3>
        <p style={{ fontSize: '1rem', lineHeight: 1.6 }}>
          Moving forward, we plan to incorporate more advanced deep learning models and extend our
          data pipeline to include additional health indicators. We also aim to develop a dedicated
          mobile application to empower users to track their health data more conveniently.
        </p>
      </section>
    </div>
  )
}

export default About
