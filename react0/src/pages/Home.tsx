import React, { useState, useEffect, ChangeEvent } from 'react'
import axios from 'axios'
import { Line } from 'react-chartjs-2'
import { Chart as ChartJS } from 'chart.js/auto'

interface IPredictData {
  Pregnancies: number
  Glucose: number
  BloodPressure: number
  SkinThickness: number
  Insulin: number
  BMI: number
  DiabetesPedigreeFunction: number
  Age: number
}

interface IEvalMetrics {
  accuracy?: number
  precision?: number
  recall?: number
  f1_score?: number
}

interface IEvalResult {
  train?: IEvalMetrics
  test?: IEvalMetrics
}

interface IResult {
  status?: string
  prediction?: number
  probability?: number
  error?: string
  metrics?: IEvalResult
}

interface ITrainProgress {
  epoch: number
  train_accuracy: number
  train_f1: number
  test_accuracy: number
  test_f1: number
}

const Home: React.FC = () => {
  const [formData, setFormData] = useState<IPredictData>({
    Pregnancies: 0,
    Glucose: 0,
    BloodPressure: 0,
    SkinThickness: 0,
    Insulin: 0,
    BMI: 0,
    DiabetesPedigreeFunction: 0,
    Age: 0
  })
  const [result, setResult] = useState<IResult>({})
  const [loading, setLoading] = useState(false)
  const [serverUrl, setServerUrl] = useState('http://localhost:5000')
  const [modelType, setModelType] = useState('logistic')
  const [searchMethod, setSearchMethod] = useState('grid')
  const [searchIters, setSearchIters] = useState(20)
  const [progressData, setProgressData] = useState<ITrainProgress[]>([])
  const [intervalId, setIntervalId] = useState<NodeJS.Timeout | null>(null)

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]:
        name === 'BMI' || name === 'DiabetesPedigreeFunction'
          ? parseFloat(value || '0')
          : parseInt(value || '0', 10)
    }))
  }

  const handleTrain = async () => {
    setLoading(true)
    setResult({})
    try {
      const { data } = await axios.post(`${serverUrl}/train`, {
        model_type: modelType,
        search_method: searchMethod,
        search_iters: searchIters
      })
      setResult({ status: data.message || 'Training Completed' })
    } catch (error: any) {
      setResult({ error: error.message || 'Training Failed' })
    } finally {
      setLoading(false)
    }
  }

  const handleEvaluate = async () => {
    setLoading(true)
    setResult({})
    try {
      const { data } = await axios.post(`${serverUrl}/evaluate`, {
        save_report: true,
        report_path: 'evaluation_report.json'
      })
      setResult({
        status: data.message,
        metrics: data.data
      })
    } catch (error: any) {
      setResult({ error: error.message || 'Evaluation Failed' })
    } finally {
      setLoading(false)
    }
  }

  const handlePredict = async () => {
    setLoading(true)
    setResult({})
    try {
      const { data } = await axios.post(`${serverUrl}/predict`, formData)
      const prediction = data?.data?.prediction ?? data?.prediction
      const probability = data?.data?.probability ?? data?.probability
      setResult({ prediction, probability })
    } catch (error: any) {
      setResult({ error: error.message || 'Prediction Failed' })
    } finally {
      setLoading(false)
    }
  }

  const handleRealtimeTrain = async () => {
    setLoading(true)
    setResult({})
    setProgressData([])
    try {
      const { data } = await axios.post(`${serverUrl}/train-realtime`, {
        epochs: 10,
        batch_size: 32
      })
      setResult({ status: data.message || 'Realtime Training Completed' })
    } catch (error: any) {
      setResult({ error: error.message || 'Realtime Training Failed' })
    } finally {
      setLoading(false)
    }
  }

  const fetchProgress = async () => {
    try {
      const { data } = await axios.get(`${serverUrl}/train-progress`)
      setProgressData(data?.progress || [])
    } catch {
    }
  }

  const startProgressMonitor = () => {
    if (!intervalId) {
      const id = setInterval(() => {
        fetchProgress()
      }, 2000)
      setIntervalId(id)
    }
  }

  const stopProgressMonitor = () => {
    if (intervalId) {
      clearInterval(intervalId)
      setIntervalId(null)
    }
  }

  const renderPrediction = () => {
    if (typeof result.prediction !== 'number') return null
    const label = result.prediction === 1 ? 'Likely Diabetes' : 'Unlikely Diabetes'
    return <p>Prediction: {label}</p>
  }

  const renderProbability = () => {
    if (typeof result.probability !== 'number') return null
    return <p>Probability: {(result.probability * 100).toFixed(2)}%</p>
  }

  const renderEvalMetrics = () => {
    if (!result.metrics) return null
    const { train, test } = result.metrics
    return (
      <div style={{ marginTop: '10px' }}>
        <h4>Evaluation Metrics</h4>
        {train && (
          <div style={{ marginBottom: '10px' }}>
            <strong>Train:</strong>
            <p>Accuracy: {train.accuracy?.toFixed(4)}</p>
            <p>Precision: {train.precision?.toFixed(4)}</p>
            <p>Recall: {train.recall?.toFixed(4)}</p>
            <p>F1 Score: {train.f1_score?.toFixed(4)}</p>
          </div>
        )}
        {test && (
          <div style={{ marginBottom: '10px' }}>
            <strong>Test:</strong>
            <p>Accuracy: {test.accuracy?.toFixed(4)}</p>
            <p>Precision: {test.precision?.toFixed(4)}</p>
            <p>Recall: {test.recall?.toFixed(4)}</p>
            <p>F1 Score: {test.f1_score?.toFixed(4)}</p>
          </div>
        )}
      </div>
    )
  }

  const chartData = {
    labels: progressData.map(item => `Epoch ${item.epoch}`),
    datasets: [
      {
        label: 'Train Accuracy',
        data: progressData.map(item => item.train_accuracy),
        borderColor: 'rgba(75,192,192,1)',
        backgroundColor: 'rgba(75,192,192,0.2)',
        fill: true
      },
      {
        label: 'Test Accuracy',
        data: progressData.map(item => item.test_accuracy),
        borderColor: 'rgba(153,102,255,1)',
        backgroundColor: 'rgba(153,102,255,0.2)',
        fill: true
      }
    ]
  }

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', fontFamily: 'Arial, sans-serif', padding: '20px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
      <div>
        <h2 style={{ marginBottom: '20px', textAlign: 'center' }}>Diabetes Detection</h2>
        <div style={{ backgroundColor: '#fff', borderRadius: '8px', boxShadow: '0 2px 5px rgba(0,0,0,0.1)', padding: '20px', marginBottom: '20px' }}>
          <div style={{ marginBottom: '20px' }}>
            <label style={{ display: 'block', marginBottom: '6px', fontWeight: 600 }}>Backend URL</label>
            <input
              type="text"
              value={serverUrl}
              onChange={e => setServerUrl(e.target.value)}
              style={{ width: '100%', padding: '8px', border: '1px solid #ccc', borderRadius: '4px' }}
            />
          </div>

          <div style={{ marginBottom: '20px' }}>
            <label style={{ display: 'block', marginBottom: '6px', fontWeight: 600 }}>Model Type</label>
            <select
              value={modelType}
              onChange={e => setModelType(e.target.value)}
              style={{ width: '100%', padding: '8px', border: '1px solid #ccc', borderRadius: '4px' }}
            >
              <option value="logistic">Logistic</option>
              <option value="rf">RandomForest</option>
              <option value="xgb">XGBoost</option>
              <option value="mlp">MLP</option>
              <option value="voting">Ensemble Voting</option>
              <option value="stacking">Ensemble Stacking</option>
            </select>
          </div>

          <div style={{ marginBottom: '20px' }}>
            <label style={{ display: 'block', marginBottom: '6px', fontWeight: 600 }}>Search Method</label>
            <select
              value={searchMethod}
              onChange={e => setSearchMethod(e.target.value)}
              style={{ width: '100%', padding: '8px', border: '1px solid #ccc', borderRadius: '4px' }}
            >
              <option value="grid">Grid Search</option>
              <option value="random">Random Search</option>
            </select>
          </div>

          <div style={{ marginBottom: '20px' }}>
            <label style={{ display: 'block', marginBottom: '6px', fontWeight: 600 }}>Search Iters (for random search)</label>
            <input
              type="number"
              value={searchIters}
              onChange={e => setSearchIters(parseInt(e.target.value, 10))}
              style={{ width: '100%', padding: '8px', border: '1px solid #ccc', borderRadius: '4px' }}
            />
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
            <div>
              <label style={{ display: 'block', marginBottom: '4px', fontWeight: 600 }}>Pregnancies</label>
              <input
                type="number"
                name="Pregnancies"
                value={formData.Pregnancies}
                onChange={handleChange}
                style={{ width: '100%', padding: '6px', border: '1px solid #ccc', borderRadius: '4px' }}
              />
            </div>
            <div>
              <label style={{ display: 'block', marginBottom: '4px', fontWeight: 600 }}>Glucose</label>
              <input
                type="number"
                name="Glucose"
                value={formData.Glucose}
                onChange={handleChange}
                style={{ width: '100%', padding: '6px', border: '1px solid #ccc', borderRadius: '4px' }}
              />
            </div>
            <div>
              <label style={{ display: 'block', marginBottom: '4px', fontWeight: 600 }}>BloodPressure</label>
              <input
                type="number"
                name="BloodPressure"
                value={formData.BloodPressure}
                onChange={handleChange}
                style={{ width: '100%', padding: '6px', border: '1px solid #ccc', borderRadius: '4px' }}
              />
            </div>
            <div>
              <label style={{ display: 'block', marginBottom: '4px', fontWeight: 600 }}>SkinThickness</label>
              <input
                type="number"
                name="SkinThickness"
                value={formData.SkinThickness}
                onChange={handleChange}
                style={{ width: '100%', padding: '6px', border: '1px solid #ccc', borderRadius: '4px' }}
              />
            </div>
            <div>
              <label style={{ display: 'block', marginBottom: '4px', fontWeight: 600 }}>Insulin</label>
              <input
                type="number"
                name="Insulin"
                value={formData.Insulin}
                onChange={handleChange}
                style={{ width: '100%', padding: '6px', border: '1px solid #ccc', borderRadius: '4px' }}
              />
            </div>
            <div>
              <label style={{ display: 'block', marginBottom: '4px', fontWeight: 600 }}>BMI</label>
              <input
                type="number"
                step="0.1"
                name="BMI"
                value={formData.BMI}
                onChange={handleChange}
                style={{ width: '100%', padding: '6px', border: '1px solid #ccc', borderRadius: '4px' }}
              />
            </div>
            <div>
              <label style={{ display: 'block', marginBottom: '4px', fontWeight: 600 }}>DiabetesPedigreeFunction</label>
              <input
                type="number"
                step="0.01"
                name="DiabetesPedigreeFunction"
                value={formData.DiabetesPedigreeFunction}
                onChange={handleChange}
                style={{ width: '100%', padding: '6px', border: '1px solid #ccc', borderRadius: '4px' }}
              />
            </div>
            <div>
              <label style={{ display: 'block', marginBottom: '4px', fontWeight: 600 }}>Age</label>
              <input
                type="number"
                name="Age"
                value={formData.Age}
                onChange={handleChange}
                style={{ width: '100%', padding: '6px', border: '1px solid #ccc', borderRadius: '4px' }}
              />
            </div>
          </div>

          <div style={{ marginTop: '20px' }}>
            <button
              onClick={handleTrain}
              style={{ marginRight: '10px', padding: '8px 16px', cursor: 'pointer', backgroundColor: '#007bff', border: 'none', color: '#fff', borderRadius: '4px', transition: 'background-color 0.3s', fontWeight: 600 }}
              disabled={loading}
            >
              {loading ? 'Training...' : 'Train'}
            </button>
            <button
              onClick={handleEvaluate}
              style={{ marginRight: '10px', padding: '8px 16px', cursor: 'pointer', backgroundColor: '#6c757d', border: 'none', color: '#fff', borderRadius: '4px', transition: 'background-color 0.3s', fontWeight: 600 }}
              disabled={loading}
            >
              {loading ? 'Evaluating...' : 'Evaluate'}
            </button>
            <button
              onClick={handlePredict}
              style={{ padding: '8px 16px', cursor: 'pointer', backgroundColor: '#28a745', border: 'none', color: '#fff', borderRadius: '4px', transition: 'background-color 0.3s', fontWeight: 600 }}
              disabled={loading}
            >
              {loading ? 'Predicting...' : 'Predict'}
            </button>
          </div>
        </div>

        <div style={{ border: '1px solid #ccc', borderRadius: '6px', padding: '10px', width: '100%', minHeight: '60px', backgroundColor: '#fff' }}>
          {loading && <p>Loading...</p>}
          {result.status && <p>Status: {result.status}</p>}
          {renderPrediction()}
          {renderProbability()}
          {renderEvalMetrics()}
          {result.error && <p style={{ color: 'red' }}>Error: {result.error}</p>}
        </div>
      </div>

      <div>
        <h3 style={{ marginBottom: '20px', textAlign: 'center' }}>Realtime Training</h3>
        <div style={{ backgroundColor: '#fff', borderRadius: '8px', boxShadow: '0 2px 5px rgba(0,0,0,0.1)', padding: '20px' }}>
          <div style={{ marginBottom: '10px' }}>
            <button
              onClick={async () => {
                setLoading(true)
                setResult({})
                setProgressData([])
                try {
                  const { data } = await axios.post(`${serverUrl}/train-realtime`, { epochs: 10, batch_size: 32 })
                  setResult({ status: data.message || 'Realtime Training Completed' })
                } catch (err: any) {
                  setResult({ error: err.message || 'Realtime Training Failed' })
                }
                setLoading(false)
              }}
              style={{ marginRight: '10px', padding: '8px 16px', cursor: 'pointer', backgroundColor: '#ff9800', border: 'none', color: '#fff', borderRadius: '4px', transition: 'background-color 0.3s', fontWeight: 600 }}
              disabled={loading}
            >
              Start Realtime Train
            </button>
            <button
              onClick={() => {
                if (!intervalId) {
                  const id = setInterval(() => {
                    fetchProgress()
                  }, 2000)
                  setIntervalId(id)
                }
              }}
              style={{ marginRight: '10px', padding: '8px 16px', cursor: 'pointer', backgroundColor: '#17a2b8', border: 'none', color: '#fff', borderRadius: '4px', transition: 'background-color 0.3s', fontWeight: 600 }}
            >
              Start Monitor
            </button>
            <button
              onClick={() => {
                if (intervalId) {
                  clearInterval(intervalId)
                  setIntervalId(null)
                }
              }}
              style={{ padding: '8px 16px', cursor: 'pointer', backgroundColor: '#dc3545', border: 'none', color: '#fff', borderRadius: '4px', transition: 'background-color 0.3s', fontWeight: 600 }}
            >
              Stop Monitor
            </button>
          </div>
          {progressData.length > 0 && (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: '20px' }}>
              <div style={{ border: '1px solid #ddd', borderRadius: '4px', padding: '10px', overflow: 'auto' }}>
                <h4>Training Progress</h4>
                <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: '10px' }}>
                  <thead>
                    <tr style={{ backgroundColor: '#f9f9f9' }}>
                      <th style={{ border: '1px solid #ccc', padding: '6px' }}>Epoch</th>
                      <th style={{ border: '1px solid #ccc', padding: '6px' }}>Train Acc</th>
                      <th style={{ border: '1px solid #ccc', padding: '6px' }}>Train F1</th>
                      <th style={{ border: '1px solid #ccc', padding: '6px' }}>Test Acc</th>
                      <th style={{ border: '1px solid #ccc', padding: '6px' }}>Test F1</th>
                    </tr>
                  </thead>
                  <tbody>
                    {progressData.map((row, idx) => (
                      <tr key={idx}>
                        <td style={{ border: '1px solid #ccc', padding: '6px' }}>{row.epoch}</td>
                        <td style={{ border: '1px solid #ccc', padding: '6px' }}>{row.train_accuracy.toFixed(4)}</td>
                        <td style={{ border: '1px solid #ccc', padding: '6px' }}>{row.train_f1.toFixed(4)}</td>
                        <td style={{ border: '1px solid #ccc', padding: '6px' }}>{row.test_accuracy.toFixed(4)}</td>
                        <td style={{ border: '1px solid #ccc', padding: '6px' }}>{row.test_f1.toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div style={{ border: '1px solid #ddd', borderRadius: '4px', padding: '10px', width: '320px', height: '320px', display: 'flex', flexDirection: 'column' }}>
                <h4 style={{ marginBottom: '10px', textAlign: 'center' }}>Accuracy Curve</h4>
                <div style={{ flex: '1 1 auto', position: 'relative' }}>
                  <Line
                    data={{
                      labels: progressData.map(item => `E${item.epoch}`),
                      datasets: [
                        {
                          label: 'Train Acc',
                          data: progressData.map(i => i.train_accuracy),
                          borderColor: 'rgba(75,192,192,1)',
                          backgroundColor: 'rgba(75,192,192,0.2)',
                          fill: true
                        },
                        {
                          label: 'Test Acc',
                          data: progressData.map(i => i.test_accuracy),
                          borderColor: 'rgba(153,102,255,1)',
                          backgroundColor: 'rgba(153,102,255,0.2)',
                          fill: true
                        }
                      ]
                    }}
                    options={{
                      responsive: false,
                      maintainAspectRatio: false,
                      scales: {
                        y: {
                          min: 0,
                          max: 1
                        }
                      }
                    }}
                    width={300}
                    height={260}
                  />
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default Home
