// Main JavaScript file for the Stock Prediction App

// Global variables
const currentChart = null
const bootstrap = window.bootstrap // Declare bootstrap variable
const Plotly = window.Plotly // Declare Plotly variable

// Utility functions
function formatCurrency(amount) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
  }).format(amount)
}

function formatDate(dateString) {
  return new Date(dateString).toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
  })
}

function showToast(message, type = "info") {
  // Create toast element
  const toast = document.createElement("div")
  toast.className = `toast align-items-center text-white bg-${type} border-0`
  toast.setAttribute("role", "alert")
  toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `

  // Add to toast container or create one
  let toastContainer = document.getElementById("toast-container")
  if (!toastContainer) {
    toastContainer = document.createElement("div")
    toastContainer.id = "toast-container"
    toastContainer.className = "toast-container position-fixed top-0 end-0 p-3"
    toastContainer.style.zIndex = "1055"
    document.body.appendChild(toastContainer)
  }

  toastContainer.appendChild(toast)

  // Show toast
  const bsToast = new bootstrap.Toast(toast)
  bsToast.show()

  // Remove toast element after it's hidden
  toast.addEventListener("hidden.bs.toast", () => {
    toast.remove()
  })
}

// Form validation
function validateStockTicker(ticker) {
  const regex = /^[A-Z]{1,5}$/
  return regex.test(ticker.toUpperCase())
}

// Chart utilities
function createPredictionChart(containerId, data) {
  const trace1 = {
    x: data.historical_dates,
    y: data.historical_prices,
    type: "scatter",
    mode: "lines+markers",
    name: "Historical Prices",
    line: {
      color: "#007bff",
      width: 2,
    },
    marker: {
      size: 4,
    },
  }

  const trace2 = {
    x: data.future_dates,
    y: data.predictions,
    type: "scatter",
    mode: "lines+markers",
    name: "Predictions",
    line: {
      color: "#28a745",
      width: 2,
      dash: "dash",
    },
    marker: {
      size: 4,
    },
  }

  const layout = {
    title: {
      text: `${data.ticker} Stock Price Prediction`,
      font: {
        size: 18,
        family: "Segoe UI, sans-serif",
      },
    },
    xaxis: {
      title: "Date",
      gridcolor: "#e9ecef",
    },
    yaxis: {
      title: "Price ($)",
      gridcolor: "#e9ecef",
    },
    showlegend: true,
    legend: {
      x: 0,
      y: 1,
      bgcolor: "rgba(255,255,255,0.8)",
    },
    plot_bgcolor: "white",
    paper_bgcolor: "white",
    margin: {
      l: 60,
      r: 30,
      t: 60,
      b: 60,
    },
  }

  const config = {
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ["pan2d", "lasso2d", "select2d"],
    displaylogo: false,
  }

  Plotly.newPlot(containerId, [trace1, trace2], layout, config)
}

// API utilities
async function makeAPIRequest(url, method = "GET", data = null) {
  const options = {
    method: method,
    headers: {
      "Content-Type": "application/json",
    },
  }

  if (data) {
    options.body = JSON.stringify(data)
  }

  try {
    const response = await fetch(url, options)
    const result = await response.json()
    return result
  } catch (error) {
    console.error("API request failed:", error)
    throw error
  }
}

// Stock ticker autocomplete (basic implementation)
function setupTickerAutocomplete() {
  const popularTickers = [
    "AAPL",
    "GOOGL",
    "MSFT",
    "AMZN",
    "TSLA",
    "META",
    "NVDA",
    "NFLX",
    "AMD",
    "INTC",
    "CRM",
    "ORCL",
    "ADBE",
    "PYPL",
    "UBER",
    "LYFT",
    "SPOT",
    "TWTR",
    "SNAP",
    "ZM",
    "DOCU",
    "SHOP",
    "SQ",
    "ROKU",
  ]

  const tickerInput = document.getElementById("ticker")
  if (!tickerInput) return

  // Create datalist for autocomplete
  const datalist = document.createElement("datalist")
  datalist.id = "ticker-suggestions"

  popularTickers.forEach((ticker) => {
    const option = document.createElement("option")
    option.value = ticker
    datalist.appendChild(option)
  })

  tickerInput.setAttribute("list", "ticker-suggestions")
  tickerInput.parentNode.appendChild(datalist)

  // Format input to uppercase
  tickerInput.addEventListener("input", function () {
    this.value = this.value.toUpperCase()
  })
}

// Initialize tooltips and popovers
function initializeBootstrapComponents() {
  // Initialize tooltips
  const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
  tooltipTriggerList.map((tooltipTriggerEl) => new bootstrap.Tooltip(tooltipTriggerEl))

  // Initialize popovers
  const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'))
  popoverTriggerList.map((popoverTriggerEl) => new bootstrap.Popover(popoverTriggerEl))
}

// Loading state management
function setLoadingState(element, isLoading, originalText = "") {
  if (isLoading) {
    element.disabled = true
    element.innerHTML = `
            <span class="spinner-border spinner-border-sm me-2" role="status"></span>
            Loading...
        `
  } else {
    element.disabled = false
    element.innerHTML = originalText
  }
}

// Local storage utilities
function saveToLocalStorage(key, data) {
  try {
    localStorage.setItem(key, JSON.stringify(data))
  } catch (error) {
    console.error("Failed to save to localStorage:", error)
  }
}

function getFromLocalStorage(key) {
  try {
    const data = localStorage.getItem(key)
    return data ? JSON.parse(data) : null
  } catch (error) {
    console.error("Failed to get from localStorage:", error)
    return null
  }
}

// Initialize app when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  initializeBootstrapComponents()
  setupTickerAutocomplete()

  // Save user preferences
  const savedModel = getFromLocalStorage("preferredModel")
  if (savedModel) {
    const modelSelect = document.getElementById("model")
    if (modelSelect) {
      modelSelect.value = savedModel
    }
  }

  // Save model preference when changed
  const modelSelect = document.getElementById("model")
  if (modelSelect) {
    modelSelect.addEventListener("change", function () {
      saveToLocalStorage("preferredModel", this.value)
    })
  }
})

// Error handling
window.addEventListener("error", (e) => {
  console.error("Global error:", e.error)
  showToast("An unexpected error occurred. Please try again.", "danger")
})

// Handle network errors
window.addEventListener("online", () => {
  showToast("Connection restored", "success")
})

window.addEventListener("offline", () => {
  showToast("Connection lost. Some features may not work.", "warning")
})
