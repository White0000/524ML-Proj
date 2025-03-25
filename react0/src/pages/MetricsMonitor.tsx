import React, { useEffect, useRef, useState, useCallback } from 'react'
import axios from 'axios'
import { Line } from 'react-chartjs-2'
import { Chart as ChartJS, ChartOptions, ChartData } from 'chart.js/auto'

interface MonitorData {
  timestamp: number
  current_value: number
}

interface MetricsMonitorProps {
  serverUrl?: string
  endpoint?: string
  title?: string
  updateIntervalMs?: number
  maxDataPoints?: number
  chartOptions?: ChartOptions<'line'>
}

ChartJS.register()

const MetricsMonitor: React.FC<MetricsMonitorProps> = ({
  serverUrl = '',
  endpoint = '/monitor',
  title = 'Real-time Metrics Monitor',
  updateIntervalMs = 5000,
  maxDataPoints = 20,
  chartOptions
}) => {
  const [historyData, setHistoryData] = useState<MonitorData[]>([])
  const [running, setRunning] = useState<boolean>(true)
  const [errorMsg, setErrorMsg] = useState<string>('')
  const intervalRef = useRef<NodeJS.Timeout | null>(null)

  const fetchData = useCallback(async () => {
    try {
      if (serverUrl && endpoint) {
        const res = await axios.get<MonitorData>(`${serverUrl}${endpoint}`)
        const data = res.data
        setHistoryData(prev => {
          const newData = [...prev, data]
          if (newData.length > maxDataPoints) {
            newData.shift()
          }
          return newData
        })
        setErrorMsg('')
      } else {
        const mock = {
          timestamp: Date.now(),
          current_value: Math.floor(Math.random() * 50) + 100
        }
        setHistoryData(prev => {
          const newData = [...prev, mock]
          if (newData.length > maxDataPoints) {
            newData.shift()
          }
          return newData
        })
      }
    } catch (error: any) {
      setErrorMsg(error.message || 'Data fetch failed')
    }
  }, [serverUrl, endpoint, maxDataPoints])

  const startMonitoring = useCallback(() => {
    if (!intervalRef.current) {
      intervalRef.current = setInterval(fetchData, updateIntervalMs)
      setRunning(true)
    }
  }, [fetchData, updateIntervalMs])

  const stopMonitoring = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
      setRunning(false)
    }
  }, [])

  useEffect(() => {
    fetchData()
    intervalRef.current = setInterval(fetchData, updateIntervalMs)
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [fetchData, updateIntervalMs])

  const data: ChartData<'line'> = {
    labels: historyData.map(item => new Date(item.timestamp).toLocaleTimeString()),
    datasets: [
      {
        label: 'Real-time Value',
        data: historyData.map(item => item.current_value),
        borderColor: 'rgba(75,192,192,1)',
        backgroundColor: 'rgba(75,192,192,0.2)',
        fill: true
      }
    ]
  }

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
      <h2 style={{ marginBottom: '10px' }}>{title}</h2>
      <p style={{ marginBottom: '20px' }}>
        The line chart below updates every {(updateIntervalMs / 1000).toFixed(1)} seconds
        {serverUrl ? ` from ${serverUrl}${endpoint}` : ' using mock data'}.
      </p>
      <div style={{ maxWidth: '800px', marginBottom: '20px' }}>
        <Line data={data} options={chartOptions} />
      </div>
      <div style={{ marginBottom: '10px' }}>
        <button
          onClick={startMonitoring}
          style={{
            marginRight: '10px',
            padding: '8px 16px',
            cursor: 'pointer',
            backgroundColor: '#007bff',
            border: 'none',
            color: '#fff',
            borderRadius: '4px'
          }}
          disabled={running}
        >
          Start
        </button>
        <button
          onClick={stopMonitoring}
          style={{
            padding: '8px 16px',
            cursor: 'pointer',
            backgroundColor: '#dc3545',
            border: 'none',
            color: '#fff',
            borderRadius: '4px'
          }}
          disabled={!running}
        >
          Stop
        </button>
      </div>
      {errorMsg && (
        <div style={{ color: 'red' }}>
          <strong>Error: </strong>{errorMsg}
        </div>
      )}
    </div>
  )
}

export default MetricsMonitor
