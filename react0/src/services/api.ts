import axios, { AxiosError, InternalAxiosRequestConfig, AxiosResponse } from 'axios'

interface TrainModelRequest { model_type?: string; [key: string]: any }
interface TrainModelResponse { code?: number; message?: string; data?: any }
interface EvaluateModelRequest { save_report?: boolean; report_path?: string; [key: string]: any }
interface EvaluateModelResponse { code?: number; message?: string; data?: any }
interface PredictRequest {
  Pregnancies: number
  Glucose: number
  BloodPressure: number
  SkinThickness: number
  Insulin: number
  BMI: number
  DiabetesPedigreeFunction: number
  Age: number
  [key: string]: any
}
interface PredictResponse {
  code?: number
  message?: string
  data?: {
    prediction?: number
    probability?: number
  }
  prediction?: number
  probability?: number
}

const apiClient = axios.create({
  baseURL: (process.env.API_URL as string) || 'http://localhost:5000',
  timeout: 10000
})

apiClient.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => config,
  (error: AxiosError) => Promise.reject(error)
)

apiClient.interceptors.response.use(
  (response: AxiosResponse) => response,
  (error: AxiosError) => Promise.reject(error)
)

export async function trainModel(payload?: TrainModelRequest): Promise<TrainModelResponse> {
  const { data } = await apiClient.post<TrainModelResponse>('/train', payload || {})
  return data
}

export async function evaluateModel(payload?: EvaluateModelRequest): Promise<EvaluateModelResponse> {
  const { data } = await apiClient.post<EvaluateModelResponse>('/evaluate', payload || {})
  return data
}

export async function predictDiabetes(payload: PredictRequest): Promise<PredictResponse> {
  const { data } = await apiClient.post<PredictResponse>('/predict', payload)
  return data
}
