import React from 'react'
import ReactDOM from 'react-dom/client'
import { Provider } from 'react-redux'
import { store } from './store'
import App from './App'
import './styles/global.css'
// 1) 变量 (CSS Variables)
import './styles/variables.css'

// 2) 基础重置 / 全局基础
import './styles/base.css'

// 3) 布局 (Header/Sidebar/Layout结构)
import './styles/layout.css'

// 4) 通用组件 (按钮、卡片、表单等)
import './styles/components.css'

// 5) 实用工具类 (margin、flex-center等)
import './styles/utilities.css'

// 6) 页面级 (Home/About/Dashboard的 .home-page 等)
import './styles/pages.css'

const container = document.getElementById('root')
if (!container) throw new Error('Root container not found')
const root = ReactDOM.createRoot(container)

root.render(
  <React.StrictMode>
    <Provider store={store}>
      <App />
    </Provider>
  </React.StrictMode>
)
