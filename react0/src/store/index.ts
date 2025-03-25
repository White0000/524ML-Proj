import { configureStore, createSlice, PayloadAction } from '@reduxjs/toolkit'

interface AppState {
  globalMessage: string
}

const initialAppState: AppState = {
  globalMessage: 'Hello from Redux Store'
}

const appSlice = createSlice({
  name: 'app',
  initialState: initialAppState,
  reducers: {
    setGlobalMessage(state, action: PayloadAction<string>) {
      state.globalMessage = action.payload
    }
  }
})

export const { setGlobalMessage } = appSlice.actions

interface TrainingState {
  isTraining: boolean
  trainingStatus: string
}

const initialTrainingState: TrainingState = {
  isTraining: false,
  trainingStatus: 'Not started'
}

const trainingSlice = createSlice({
  name: 'training',
  initialState: initialTrainingState,
  reducers: {
    startTraining(state) {
      state.isTraining = true
      state.trainingStatus = 'In progress'
    },
    completeTraining(state) {
      state.isTraining = false
      state.trainingStatus = 'Completed'
    },
    failTraining(state) {
      state.isTraining = false
      state.trainingStatus = 'Failed'
    }
  }
})

export const { startTraining, completeTraining, failTraining } = trainingSlice.actions

export const store = configureStore({
  reducer: {
    app: appSlice.reducer,
    training: trainingSlice.reducer
  }
})

export type RootState = ReturnType<typeof store.getState>
export type AppDispatch = typeof store.dispatch
