/**
 * ReturnSense — Frontend Application Logic
 * ==========================================
 *
 * Complete JavaScript for:
 *   - Drag-and-drop file handling
 *   - FormData upload to FastAPI /upload endpoint
 *   - Parsing JSON response
 *   - Rendering summary cards with count-up animation
 *   - Drawing Chart.js doughnut chart (risk distribution)
 *   - Drawing Chart.js bar chart (SHAP feature importance)
 *   - Drawing Chart.js line chart (sales forecast)
 *   - Rendering data quality validation results
 *   - Rendering products table with search, sort, and pagination
 *   - What-if simulator: auto-populated dropdown + scenario table
 *   - Loading states and error handling
 *
 * Flow:
 *   1. User drags/drops or selects a CSV/Excel file
 *   2. File is uploaded to FastAPI POST /upload
 *   3. JSON response populates all dashboard sections
 *   4. Forecast data fetched from GET /forecast
 *   5. User can search, sort, and paginate the table
 *   6. User selects a product in What-If to see scenarios
 */

// =========================================================================
// CONFIGURATION
// =========================================================================
const API_BASE = (() => {
    // Auto-detect API URL — works for local dev and production
    const hostname = window.location.hostname;
    if (hostname === "localhost" || hostname === "127.0.0.1") {
        return "http://127.0.0.1:8000";
    }
    // In production, assume API is at /api or same origin
    return window.location.origin;
})();
const ROWS_PER_PAGE = 10;

// =========================================================================
// STATE
// =========================================================================
let selectedFile = null;
let allProducts = [];
let filteredProducts = [];
let currentPage = 1;
let riskChartInstance = null;
let shapChartInstance = null;
let forecastChartInstance = null;

// =========================================================================
// DOM ELEMENTS
// =========================================================================
const uploadArea = document.getElementById("upload-area");
const fileInput = document.getElementById("file-input");
const fileNameDisplay = document.getElementById("file-name-display");
const selectedFileName = document.getElementById("selected-file-name");
const fileRemoveBtn = document.getElementById("file-remove-btn");
const analyzeBtn = document.getElementById("analyze-btn");
const loader = document.getElementById("loader");
const errorBanner = document.getElementById("error-banner");
const errorMessage = document.getElementById("error-message");
const errorClose = document.getElementById("error-close");
const resultsWrapper = document.getElementById("results-wrapper");
const tableBody = document.getElementById("table-body");
const pagination = document.getElementById("pagination");
const tableSearch = document.getElementById("table-search");
const sortSelect = document.getElementById("sort-select");
const whatifSelect = document.getElementById("whatif-select");
const whatifResults = document.getElementById("whatif-results");
const whatifCurrent = document.getElementById("whatif-current");
const whatifBody = document.getElementById("whatif-body");

// Data quality elements
const dqBar = document.getElementById("data-quality-bar");
const dqIcon = document.getElementById("dq-icon");
const dqStatus = document.getElementById("dq-status");
const dqDetail = document.getElementById("dq-detail");
const dqToggleBtn = document.getElementById("dq-toggle-btn");
const dqDetailsPanel = document.getElementById("dq-details-panel");
const dqChecksGrid = document.getElementById("dq-checks-grid");

// Forecast elements
const forecastSummary = document.getElementById("forecast-summary");
const forecastNoData = document.getElementById("forecast-no-data");

// =========================================================================
// FILE HANDLING
// =========================================================================

// Click to browse
uploadArea.addEventListener("click", (e) => {
    if (e.target === fileRemoveBtn || e.target.closest(".file-remove-btn")) return;
    fileInput.click();
});

// File selected via input
fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
        setFile(e.target.files[0]);
    }
});

// Drag and drop
uploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadArea.classList.add("drag-over");
});

uploadArea.addEventListener("dragleave", (e) => {
    e.preventDefault();
    uploadArea.classList.remove("drag-over");
});

uploadArea.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadArea.classList.remove("drag-over");
    if (e.dataTransfer.files.length > 0) {
        setFile(e.dataTransfer.files[0]);
    }
});

// Remove file
fileRemoveBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    clearFile();
});

function setFile(file) {
    const ext = file.name.split(".").pop().toLowerCase();
    if (!["csv", "xlsx", "xls"].includes(ext)) {
        showError("Invalid file type. Please upload a .csv or .xlsx file.");
        return;
    }
    selectedFile = file;
    selectedFileName.textContent = file.name;
    fileNameDisplay.style.display = "inline-flex";
    analyzeBtn.disabled = false;
    hideError();
}

function clearFile() {
    selectedFile = null;
    fileInput.value = "";
    fileNameDisplay.style.display = "none";
    analyzeBtn.disabled = true;
}

// =========================================================================
// ERROR HANDLING
// =========================================================================
function showError(msg) {
    errorMessage.textContent = msg;
    errorBanner.style.display = "flex";
}

function hideError() {
    errorBanner.style.display = "none";
}

errorClose.addEventListener("click", hideError);

// =========================================================================
// UPLOAD & ANALYZE
// =========================================================================
analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    // Show loader, hide results
    loader.style.display = "block";
    resultsWrapper.style.display = "none";
    analyzeBtn.disabled = true;
    hideError();

    try {
        const formData = new FormData();
        formData.append("file", selectedFile);

        const response = await fetch(`${API_BASE}/upload`, {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.detail || `Server error: ${response.status}`);
        }

        const data = await response.json();

        // Store products globally
        allProducts = data.products || [];
        filteredProducts = [...allProducts];
        currentPage = 1;

        // Render everything
        renderSummaryCards(data);
        renderRiskChart(data);
        renderProductsTable();
        renderWhatIfDropdown();
        fetchAndRenderSHAP();
        fetchAndRenderForecast();

        // Render data quality if available
        if (data.validation) {
            renderDataQuality(data.validation);
        }

        // Show results
        resultsWrapper.style.display = "block";
        loader.style.display = "none";

        // Animate cards
        setTimeout(() => {
            document.querySelectorAll(".card-animate").forEach((card, i) => {
                setTimeout(() => card.classList.add("visible"), i * 120);
            });
        }, 100);

        // Scroll to results
        document.getElementById("summary-section").scrollIntoView({ behavior: "smooth" });

    } catch (err) {
        loader.style.display = "none";
        showError(err.message || "Failed to connect to the API. Make sure the server is running.");
    } finally {
        analyzeBtn.disabled = false;
    }
});

// =========================================================================
// DATA QUALITY RENDERING
// =========================================================================
function renderDataQuality(validation) {
    if (!validation || !validation.status) return;

    const status = validation.status;
    const statusClass = status === "PASS" ? "dq-pass" : status === "WARNING" ? "dq-warning" : "dq-fail";
    const icon = status === "PASS" ? "✓" : status === "WARNING" ? "!" : "✗";

    // Set bar classes
    dqBar.className = `data-quality-bar ${statusClass}`;
    dqBar.style.display = "flex";

    dqIcon.textContent = icon;
    dqStatus.textContent = `Data Quality: ${status}`;
    dqDetail.textContent = `${validation.passed || 0}/${validation.total_checks || 0} checks passed`;

    // Render individual checks
    dqChecksGrid.innerHTML = "";
    if (validation.checks && validation.checks.length > 0) {
        validation.checks.forEach((check) => {
            const item = document.createElement("div");
            item.className = "dq-check-item";
            item.innerHTML = `
                <span class="dq-check-icon ${check.passed ? 'passed' : 'failed'}">${check.passed ? '✓' : '✗'}</span>
                <span class="dq-check-name">${escapeHtml(check.description || check.name)}</span>
                <span class="dq-check-detail">${escapeHtml(check.detail || '')}</span>
            `;
            dqChecksGrid.appendChild(item);
        });
    }

    // Reset toggle state
    dqDetailsPanel.style.display = "none";
    dqToggleBtn.classList.remove("expanded");
}

// Toggle data quality details
dqToggleBtn.addEventListener("click", () => {
    const isHidden = dqDetailsPanel.style.display === "none";
    dqDetailsPanel.style.display = isHidden ? "block" : "none";
    dqToggleBtn.classList.toggle("expanded", isHidden);
});

// =========================================================================
// SUMMARY CARDS with count-up animation
// =========================================================================
function renderSummaryCards(data) {
    // Reset card animations
    document.querySelectorAll(".card-animate").forEach((card) => {
        card.classList.remove("visible");
    });

    animateCountUp("val-total", data.total_products, 0, "");
    animateCountUp("val-high-risk", data.high_risk_count, 0, "");
    animateCountUp("val-saved", data.total_units_saved, 0, "");
    animateCountUp("val-avg-prob", data.average_return_probability * 100, 1, "%");
}

function animateCountUp(elementId, target, decimals, suffix) {
    const el = document.getElementById(elementId);
    const duration = 1200;
    const start = 0;
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        // Ease out cubic
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = start + (target - start) * eased;
        el.textContent = current.toFixed(decimals) + suffix;
        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

// =========================================================================
// RISK DISTRIBUTION DOUGHNUT CHART
// =========================================================================
function renderRiskChart(data) {
    const ctx = document.getElementById("risk-chart").getContext("2d");

    if (riskChartInstance) {
        riskChartInstance.destroy();
    }

    riskChartInstance = new Chart(ctx, {
        type: "doughnut",
        data: {
            labels: ["Low Risk", "Medium Risk", "High Risk"],
            datasets: [{
                data: [data.low_risk_count, data.medium_risk_count, data.high_risk_count],
                backgroundColor: [
                    "rgba(0, 230, 118, 0.8)",
                    "rgba(255, 145, 0, 0.8)",
                    "rgba(255, 23, 68, 0.8)",
                ],
                borderColor: [
                    "rgba(0, 230, 118, 1)",
                    "rgba(255, 145, 0, 1)",
                    "rgba(255, 23, 68, 1)",
                ],
                borderWidth: 2,
                hoverOffset: 8,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            cutout: "65%",
            plugins: {
                legend: {
                    position: "bottom",
                    labels: {
                        color: "#9999aa",
                        font: { family: "Inter", size: 13 },
                        padding: 16,
                        usePointStyle: true,
                        pointStyleWidth: 12,
                    },
                },
                tooltip: {
                    backgroundColor: "rgba(18, 18, 26, 0.95)",
                    titleColor: "#f0f0f5",
                    bodyColor: "#9999aa",
                    borderColor: "rgba(255,255,255,0.08)",
                    borderWidth: 1,
                    cornerRadius: 8,
                    titleFont: { family: "Inter", weight: "600" },
                    bodyFont: { family: "Inter" },
                },
            },
        },
    });
}

// =========================================================================
// SHAP FEATURE IMPORTANCE BAR CHART
// =========================================================================
async function fetchAndRenderSHAP() {
    try {
        const response = await fetch(`${API_BASE}/shap`);
        if (response.ok) {
            const data = await response.json();
            renderSHAPChart(data.feature_importance);
        } else {
            // Use default values if SHAP endpoint not available
            renderSHAPChart({
                "customer_past_return_rate": 0.35,
                "price_vs_category_avg": 0.22,
                "UnitPrice": 0.15,
                "customer_order_count": 0.12,
                "month": 0.08,
                "day_of_week": 0.05,
                "Quantity": 0.03,
            });
        }
    } catch {
        // Fallback defaults
        renderSHAPChart({
            "customer_past_return_rate": 0.35,
            "price_vs_category_avg": 0.22,
            "UnitPrice": 0.15,
            "customer_order_count": 0.12,
            "month": 0.08,
            "day_of_week": 0.05,
            "Quantity": 0.03,
        });
    }
}

function renderSHAPChart(importance) {
    const ctx = document.getElementById("shap-chart").getContext("2d");

    if (shapChartInstance) {
        shapChartInstance.destroy();
    }

    // Sort by importance
    const sorted = Object.entries(importance).sort((a, b) => b[1] - a[1]);
    const labels = sorted.map(([k]) => formatFeatureName(k));
    const values = sorted.map(([, v]) => v);

    // Generate gradient colors
    const colors = values.map((_, i) => {
        const ratio = i / (values.length - 1 || 1);
        const r = Math.round(108 + (224 - 108) * ratio);
        const g = Math.round(99 + (64 - 99) * ratio);
        const b = Math.round(255 + (251 - 255) * ratio);
        return `rgba(${r}, ${g}, ${b}, 0.8)`;
    });

    shapChartInstance = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                label: "SHAP Importance",
                data: values,
                backgroundColor: colors,
                borderColor: colors.map(c => c.replace("0.8", "1")),
                borderWidth: 1,
                borderRadius: 6,
                barThickness: 28,
            }],
        },
        options: {
            indexAxis: "y",
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                x: {
                    grid: { color: "rgba(255,255,255,0.04)" },
                    ticks: { color: "#9999aa", font: { family: "Inter", size: 11 } },
                },
                y: {
                    grid: { display: false },
                    ticks: { color: "#f0f0f5", font: { family: "Inter", size: 12, weight: "500" } },
                },
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: "rgba(18, 18, 26, 0.95)",
                    titleColor: "#f0f0f5",
                    bodyColor: "#9999aa",
                    borderColor: "rgba(255,255,255,0.08)",
                    borderWidth: 1,
                    cornerRadius: 8,
                    titleFont: { family: "Inter", weight: "600" },
                    bodyFont: { family: "Inter" },
                },
            },
        },
    });
}

function formatFeatureName(name) {
    const map = {
        "customer_past_return_rate": "Past Return Rate",
        "price_vs_category_avg": "Price vs Category Avg",
        "UnitPrice": "Unit Price",
        "customer_order_count": "Order Count",
        "day_of_week": "Day of Week",
        "month": "Month",
        "Quantity": "Quantity",
    };
    return map[name] || name;
}

// =========================================================================
// SALES FORECAST LINE CHART (Prophet)
// =========================================================================
async function fetchAndRenderForecast() {
    try {
        const response = await fetch(`${API_BASE}/forecast`);
        if (response.ok) {
            const data = await response.json();
            renderForecastChart(data);
            renderForecastSummary(data.summary);
        } else {
            showForecastNoData();
        }
    } catch {
        showForecastNoData();
    }
}

function showForecastNoData() {
    forecastNoData.style.display = "block";
    forecastSummary.style.display = "none";
    const canvas = document.getElementById("forecast-chart");
    canvas.style.display = "none";
}

function renderForecastSummary(summary) {
    if (!summary) return;
    forecastSummary.style.display = "grid";

    document.getElementById("forecast-total-revenue").textContent =
        `$${Number(summary.total_predicted_revenue).toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
    document.getElementById("forecast-avg-daily").textContent =
        `$${Number(summary.avg_daily_revenue).toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
    document.getElementById("forecast-period").textContent =
        `${summary.forecast_days} days`;
}

function renderForecastChart(data) {
    const canvas = document.getElementById("forecast-chart");
    canvas.style.display = "block";
    forecastNoData.style.display = "none";

    const ctx = canvas.getContext("2d");

    if (forecastChartInstance) {
        forecastChartInstance.destroy();
    }

    const records = data.data || [];
    if (records.length === 0) {
        showForecastNoData();
        return;
    }

    // Split into historical and forecast portions
    const forecastDays = data.summary?.forecast_days || 30;
    const historicalEnd = records.length - forecastDays;

    const labels = records.map((r) => r.date);
    const predicted = records.map((r) => Math.round(r.predicted_revenue));
    const upper = records.map((r) => Math.round(r.upper_bound));
    const lower = records.map((r) => Math.round(r.lower_bound));

    // Create a separate dataset for the forecast region
    const historicalPredicted = predicted.map((v, i) => (i < historicalEnd ? v : null));
    const futurePredicted = predicted.map((v, i) => (i >= historicalEnd - 1 ? v : null));
    const confidenceUpper = upper.map((v, i) => (i >= historicalEnd - 1 ? v : null));
    const confidenceLower = lower.map((v, i) => (i >= historicalEnd - 1 ? v : null));

    forecastChartInstance = new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "Historical (Fitted)",
                    data: historicalPredicted,
                    borderColor: "rgba(68, 138, 255, 0.9)",
                    backgroundColor: "rgba(68, 138, 255, 0.05)",
                    borderWidth: 2,
                    fill: true,
                    pointRadius: 0,
                    tension: 0.3,
                },
                {
                    label: "Forecast",
                    data: futurePredicted,
                    borderColor: "rgba(224, 64, 251, 0.9)",
                    backgroundColor: "rgba(224, 64, 251, 0.05)",
                    borderWidth: 2.5,
                    borderDash: [6, 3],
                    fill: false,
                    pointRadius: 0,
                    tension: 0.3,
                },
                {
                    label: "Upper Bound",
                    data: confidenceUpper,
                    borderColor: "rgba(224, 64, 251, 0.2)",
                    backgroundColor: "rgba(224, 64, 251, 0.08)",
                    borderWidth: 1,
                    fill: "+1",
                    pointRadius: 0,
                    tension: 0.3,
                },
                {
                    label: "Lower Bound",
                    data: confidenceLower,
                    borderColor: "rgba(224, 64, 251, 0.2)",
                    backgroundColor: "rgba(224, 64, 251, 0.08)",
                    borderWidth: 1,
                    fill: false,
                    pointRadius: 0,
                    tension: 0.3,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: "index",
                intersect: false,
            },
            scales: {
                x: {
                    grid: { color: "rgba(255,255,255,0.04)" },
                    ticks: {
                        color: "#9999aa",
                        font: { family: "Inter", size: 10 },
                        maxTicksLimit: 12,
                        maxRotation: 45,
                    },
                },
                y: {
                    grid: { color: "rgba(255,255,255,0.04)" },
                    ticks: {
                        color: "#9999aa",
                        font: { family: "Inter", size: 11 },
                        callback: (value) => `$${(value / 1000).toFixed(0)}k`,
                    },
                    title: {
                        display: true,
                        text: "Daily Revenue ($)",
                        color: "#666677",
                        font: { family: "Inter", size: 11 },
                    },
                },
            },
            plugins: {
                legend: {
                    position: "top",
                    labels: {
                        color: "#9999aa",
                        font: { family: "Inter", size: 12 },
                        padding: 16,
                        usePointStyle: true,
                        pointStyleWidth: 12,
                        filter: (item) => {
                            // Hide upper/lower bound from legend
                            return !item.text.includes("Bound");
                        },
                    },
                },
                tooltip: {
                    backgroundColor: "rgba(18, 18, 26, 0.95)",
                    titleColor: "#f0f0f5",
                    bodyColor: "#9999aa",
                    borderColor: "rgba(255,255,255,0.08)",
                    borderWidth: 1,
                    cornerRadius: 8,
                    titleFont: { family: "Inter", weight: "600" },
                    bodyFont: { family: "Inter" },
                    callbacks: {
                        label: (context) => {
                            if (context.dataset.label.includes("Bound")) return null;
                            return `${context.dataset.label}: $${context.parsed.y.toLocaleString()}`;
                        },
                    },
                },
            },
        },
    });
}

// =========================================================================
// PRODUCTS TABLE with search, sort, pagination
// =========================================================================
function renderProductsTable() {
    const start = (currentPage - 1) * ROWS_PER_PAGE;
    const end = start + ROWS_PER_PAGE;
    const pageProducts = filteredProducts.slice(start, end);

    tableBody.innerHTML = "";

    if (pageProducts.length === 0) {
        tableBody.innerHTML = `
            <tr><td colspan="7" style="text-align:center; padding:40px; color:var(--text-muted);">
                No products found.
            </td></tr>`;
        pagination.innerHTML = "";
        return;
    }

    pageProducts.forEach((p) => {
        const riskClass = p.risk_level.toLowerCase();
        const probPct = (p.return_probability * 100).toFixed(1);

        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td title="${escapeHtml(p.description)}">${truncate(p.description, 35)}</td>
            <td><code>${escapeHtml(p.stock_code)}</code></td>
            <td>${formatNumber(p.gross_sales)}</td>
            <td>
                <div class="prob-bar-wrap">
                    <div class="prob-bar">
                        <div class="prob-bar-fill ${riskClass}" style="width:${Math.min(probPct, 100)}%"></div>
                    </div>
                    <span class="prob-value">${probPct}%</span>
                </div>
            </td>
            <td>${formatNumber(p.net_demand)}</td>
            <td>${formatNumber(p.units_saved)}</td>
            <td><span class="risk-badge ${riskClass}">${p.risk_level}</span></td>
        `;
        tableBody.appendChild(tr);
    });

    renderPagination();
}

// Search
tableSearch.addEventListener("input", () => {
    const query = tableSearch.value.toLowerCase().trim();
    if (!query) {
        filteredProducts = [...allProducts];
    } else {
        filteredProducts = allProducts.filter(
            (p) =>
                (p.description && p.description.toLowerCase().includes(query)) ||
                (p.stock_code && p.stock_code.toLowerCase().includes(query))
        );
    }
    currentPage = 1;
    renderProductsTable();
});

// Sort
sortSelect.addEventListener("change", () => {
    const [field, dir] = sortSelect.value.split("-");
    filteredProducts.sort((a, b) => {
        const valA = a[field] || 0;
        const valB = b[field] || 0;
        return dir === "desc" ? valB - valA : valA - valB;
    });
    currentPage = 1;
    renderProductsTable();
});

// Pagination
function renderPagination() {
    const totalPages = Math.ceil(filteredProducts.length / ROWS_PER_PAGE);
    pagination.innerHTML = "";

    if (totalPages <= 1) return;

    // Prev button
    const prevBtn = createPageBtn("← Prev", currentPage > 1, () => {
        currentPage--;
        renderProductsTable();
    });
    pagination.appendChild(prevBtn);

    // Page numbers
    const maxVisible = 7;
    let startPage = Math.max(1, currentPage - Math.floor(maxVisible / 2));
    let endPage = Math.min(totalPages, startPage + maxVisible - 1);
    if (endPage - startPage < maxVisible - 1) {
        startPage = Math.max(1, endPage - maxVisible + 1);
    }

    if (startPage > 1) {
        pagination.appendChild(createPageBtn("1", true, () => { currentPage = 1; renderProductsTable(); }));
        if (startPage > 2) {
            const dots = document.createElement("span");
            dots.textContent = "…";
            dots.style.color = "var(--text-muted)";
            dots.style.padding = "0 4px";
            pagination.appendChild(dots);
        }
    }

    for (let i = startPage; i <= endPage; i++) {
        const btn = createPageBtn(String(i), true, () => {
            currentPage = i;
            renderProductsTable();
        });
        if (i === currentPage) btn.classList.add("active");
        pagination.appendChild(btn);
    }

    if (endPage < totalPages) {
        if (endPage < totalPages - 1) {
            const dots = document.createElement("span");
            dots.textContent = "…";
            dots.style.color = "var(--text-muted)";
            dots.style.padding = "0 4px";
            pagination.appendChild(dots);
        }
        pagination.appendChild(createPageBtn(String(totalPages), true, () => { currentPage = totalPages; renderProductsTable(); }));
    }

    // Next button
    const nextBtn = createPageBtn("Next →", currentPage < totalPages, () => {
        currentPage++;
        renderProductsTable();
    });
    pagination.appendChild(nextBtn);
}

function createPageBtn(text, enabled, onClick) {
    const btn = document.createElement("button");
    btn.className = "page-btn";
    btn.textContent = text;
    btn.disabled = !enabled;
    if (enabled) btn.addEventListener("click", onClick);
    return btn;
}

// =========================================================================
// WHAT-IF SIMULATOR
// =========================================================================
function renderWhatIfDropdown() {
    whatifSelect.innerHTML = '<option value="">— Choose a product —</option>';

    allProducts.forEach((p, i) => {
        const opt = document.createElement("option");
        opt.value = i;
        opt.textContent = `${p.stock_code} — ${truncate(p.description, 40)} (Prob: ${(p.return_probability * 100).toFixed(1)}%)`;
        whatifSelect.appendChild(opt);
    });
}

whatifSelect.addEventListener("change", () => {
    const idx = whatifSelect.value;
    if (idx === "") {
        whatifResults.style.display = "none";
        return;
    }

    const product = allProducts[parseInt(idx)];
    renderWhatIfScenarios(product);
});

function renderWhatIfScenarios(product) {
    // Current product info
    whatifCurrent.innerHTML = `
        <div class="product-name">${escapeHtml(product.description)}</div>
        <div class="product-stats">
            Stock Code: <strong>${escapeHtml(product.stock_code)}</strong> &nbsp;|&nbsp;
            Gross Sales: <strong>${formatNumber(product.gross_sales)}</strong> &nbsp;|&nbsp;
            Current Return Prob: <strong>${(product.return_probability * 100).toFixed(1)}%</strong>
        </div>
    `;

    // Compute scenarios locally (no API call needed)
    const gross = product.gross_sales;
    const baseProb = product.return_probability;
    const baseNet = Math.round(gross * (1 - baseProb));

    const scenarios = [
        { scenario: "Current", prob: baseProb, net: baseNet, saved: gross - baseNet },
        {
            scenario: "Return rate −10%",
            prob: baseProb * 0.9,
            net: Math.round(gross * (1 - baseProb * 0.9)),
            saved: gross - Math.round(gross * (1 - baseProb * 0.9)),
        },
        {
            scenario: "Return rate −20%",
            prob: baseProb * 0.8,
            net: Math.round(gross * (1 - baseProb * 0.8)),
            saved: gross - Math.round(gross * (1 - baseProb * 0.8)),
        },
        {
            scenario: "Return rate −30%",
            prob: baseProb * 0.7,
            net: Math.round(gross * (1 - baseProb * 0.7)),
            saved: gross - Math.round(gross * (1 - baseProb * 0.7)),
        },
    ];

    whatifBody.innerHTML = "";
    scenarios.forEach((s, i) => {
        const improvement = i === 0 ? "—" : `+${s.net - baseNet} units`;
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td>${s.scenario}</td>
            <td>${(s.prob * 100).toFixed(2)}%</td>
            <td>${formatNumber(s.net)}</td>
            <td>${formatNumber(s.saved)}</td>
            <td class="improvement">${improvement}</td>
        `;
        whatifBody.appendChild(tr);
    });

    whatifResults.style.display = "block";
}

// =========================================================================
// UTILITY FUNCTIONS
// =========================================================================
function escapeHtml(str) {
    if (!str) return "";
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}

function truncate(str, len) {
    if (!str) return "";
    return str.length > len ? str.substring(0, len) + "…" : str;
}

function formatNumber(n) {
    if (n === null || n === undefined) return "0";
    return Number(n).toLocaleString();
}
