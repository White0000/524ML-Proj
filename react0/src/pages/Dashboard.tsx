import React, { useState, useEffect } from 'react'
import axios from 'axios'
import { Line, Pie, Bar } from 'react-chartjs-2'
import { Chart as ChartJS, Title, Tooltip, Legend, CategoryScale, LinearScale, PointElement, LineElement, ArcElement, BarElement } from 'chart.js'

ChartJS.register(Title, Tooltip, Legend, CategoryScale, LinearScale, PointElement, LineElement, ArcElement, BarElement)

interface DashboardData {
  glucoseTrends?: number[]
  outcomeDistribution?: number[]
  barMetrics?: number[]
}

const Dashboard: React.FC = () => {
  const [glucoseData, setGlucoseData] = useState<number[]>([])
  const [outcomeData, setOutcomeData] = useState<number[]>([])
  const [randomBarData, setRandomBarData] = useState<number[]>([])
  const [labels, setLabels] = useState<string[]>(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'])
  const [error, setError] = useState<string>('')

  const fetchDashboardData = async () => {
    try {
      const res = await axios.get<DashboardData>('http://localhost:5000/dashboard-data')
      setGlucoseData(res.data.glucoseTrends || [])
      setOutcomeData(res.data.outcomeDistribution || [])
      setRandomBarData(res.data.barMetrics || [])
      setError('')
    } catch (e: any) {
      setError(e.message || 'Failed to fetch data')
    }
  }

  useEffect(() => {
    fetchDashboardData()
    const timer = setInterval(fetchDashboardData, 6000)
    return () => clearInterval(timer)
  }, [])

  const lineData = {
    labels,
    datasets: [
      {
        label: 'Glucose (mg/dL)',
        data: glucoseData,
        borderColor: 'rgba(75, 192, 192, 1)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        fill: true
      }
    ]
  }

  const pieData = {
    labels: ['Positive', 'Negative'],
    datasets: [
      {
        label: 'Diabetes Outcome',
        data: outcomeData,
        backgroundColor: ['rgba(255, 99, 132, 0.6)', 'rgba(53, 162, 235, 0.6)']
      }
    ]
  }

  const barData = {
    labels: ['Metric A', 'Metric B', 'Metric C', 'Metric D', 'Metric E'],
    datasets: [
      {
        label: 'Patient Metrics',
        data: randomBarData,
        backgroundColor: 'rgba(153, 102, 255, 0.6)'
      }
    ]
  }

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
      <h2 style={{ marginBottom: '20px' }}>Dashboard</h2>
      {error && (
        <p style={{ color: 'red', marginBottom: '20px' }}>{error}</p>
      )}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
        gap: '20px'
      }}>
        <div style={{
          backgroundColor: '#fff',
          borderRadius: '6px',
          boxShadow: '0 2px 5px rgba(0,0,0,0.1)',
          padding: '1rem'
        }}>
          <h4 style={{ marginBottom: '10px' }}>Monthly Glucose Trend</h4>
          <Line data={lineData} />
        </div>
        <div style={{
          backgroundColor: '#fff',
          borderRadius: '6px',
          boxShadow: '0 2px 5px rgba(0,0,0,0.1)',
          padding: '1rem'
        }}>
          <h4 style={{ marginBottom: '10px' }}>Outcome Distribution</h4>
          <Pie data={pieData} />
        </div>
        <div style={{
          backgroundColor: '#fff',
          borderRadius: '6px',
          boxShadow: '0 2px 5px rgba(0,0,0,0.1)',
          padding: '1rem'
        }}>
          <h4 style={{ marginBottom: '10px' }}>Random Patient Metrics</h4>
          <Bar data={barData} />
        </div>
        <div style={{
          backgroundColor: '#f9f9f9',
          borderRadius: '8px',
          padding: '1rem',
          boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
        }}>
          <h4 style={{ marginBottom: '10px' }}>Info</h4>
          <p style={{ margin: 0 }}>
            The charts fetch data every 6 seconds from /dashboard-data.
            The line chart shows glucose trends, the pie chart shows outcome distribution,
            and the bar chart represents patient metrics. Data originates from the latest model or database.
          </p>
        </div>
      </div>
    </div>
  )
}

export default Dashboard
