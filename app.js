const $ = s => document.querySelector(s), $$ = s => document.querySelectorAll(s);
let tx = [], summary = null, forecast = null, chart = null, user = null, mlStatus = null;
let isDemoMode = false;
const money = n => '₹' + Number(n || 0).toLocaleString('en-IN', { maximumFractionDigits: 0 });

const toast = (msg, error = false) => {
    const e = document.createElement('div');
    e.className = 'toast' + (error ? ' error' : '');
    e.textContent = msg;
    $('#toast').appendChild(e);
    setTimeout(() => e.remove(), 3500);
};

async function api(url, opt = {}) {
    const r = await fetch(url, { credentials: 'include', ...opt });
    let d = {};
    try { d = await r.json(); } catch {}
    if (!r.ok) throw new Error(d.detail || 'Something went wrong');
    return d;
}

function initials(name) {
    return name.split(/\s+/).map(x => x[0]).slice(0, 2).join('').toUpperCase();
}

function showAuth(signup = false) {
    $('#auth').classList.remove('hidden');
    $('#app').classList.add('hidden');
    $('#loginView').classList.toggle('hidden', signup);
    $('#signupView').classList.toggle('hidden', !signup);
}

$('#showSignup').onclick = () => showAuth(true);
$('#showLogin').onclick = () => showAuth(false);

// Demo Mode Triggers
function startDemoMode() {
    isDemoMode = true;
    user = {
        name: 'Demo Guide',
        email: 'demo@college.guide'
    };
    $('#auth').classList.add('hidden');
    $('#app').classList.remove('hidden');
    $('#demoBadge').classList.remove('hidden');
    $('#demoBanner').classList.remove('hidden');
    $('#exitDemo').classList.remove('hidden');
    $('#headerExitDemo').classList.remove('hidden');
    $('#helloName').textContent = 'Guide (Demo)';
    $('#sideName').textContent = 'Demo Explorer';
    $('#sideEmail').textContent = 'Personal_Finance_Dataset.csv';
    $('#avatar').textContent = 'DG';
    $('#topAvatar').textContent = 'DG';
    $('#monthLabel').textContent = 'December 2024 (Demo)';
    refresh();
    toast('Entered Demo Mode (Read-Only)');
}

function exitDemoMode() {
    isDemoMode = false;
    user = null;
    tx = [];
    summary = null;
    forecast = null;
    if (chart) {
        chart.destroy();
        chart = null;
    }
    $('#demoBadge').classList.add('hidden');
    $('#demoBanner').classList.add('hidden');
    $('#exitDemo').classList.add('hidden');
    $('#headerExitDemo').classList.add('hidden');
    $('#monthLabel').textContent = 'October 2026';
    showAuth(false);
    toast('Exited Demo Mode');
}

const btnTryDemo = $('#btnTryDemo');
if (btnTryDemo) btnTryDemo.onclick = startDemoMode;
const btnTryDemoSignup = $('#btnTryDemoSignup');
if (btnTryDemoSignup) btnTryDemoSignup.onclick = startDemoMode;

const exitDemo = $('#exitDemo');
if (exitDemo) exitDemo.onclick = exitDemoMode;
const headerExitDemo = $('#headerExitDemo');
if (headerExitDemo) headerExitDemo.onclick = exitDemoMode;
const bannerExitDemo = $('#bannerExitDemo');
if (bannerExitDemo) bannerExitDemo.onclick = exitDemoMode;

$('#loginForm').onsubmit = async e => {
    e.preventDefault();
    try {
        const d = await api('/api/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email: $('#loginEmail').value, password: $('#loginPassword').value })
        });
        isDemoMode = false;
        user = d.user;
        startApp();
    } catch (x) {
        toast(x.message, true);
    }
};

$('#signupForm').onsubmit = async e => {
    e.preventDefault();
    try {
        const d = await api('/api/auth/signup', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: $('#signupName').value, email: $('#signupEmail').value, password: $('#signupPassword').value })
        });
        isDemoMode = false;
        user = d.user;
        startApp();
    } catch (x) {
        toast(x.message, true);
    }
};

$('#logout').onclick = async () => {
    if (isDemoMode) {
        exitDemoMode();
        return;
    }
    await api('/api/auth/logout', { method: 'POST' });
    location.reload();
};

async function startApp() {
    if (isDemoMode) {
        startDemoMode();
        return;
    }
    if (!user) {
        try { user = (await api('/api/auth/me')).user; } catch { return; }
    }
    $('#demoBadge').classList.add('hidden');
    $('#demoBanner').classList.add('hidden');
    $('#exitDemo').classList.add('hidden');
    $('#headerExitDemo').classList.add('hidden');
    $('#auth').classList.add('hidden');
    $('#app').classList.remove('hidden');
    $('#helloName').textContent = user.name.split(' ')[0];
    $('#sideName').textContent = user.name;
    $('#sideEmail').textContent = user.email;
    $('#settingsEmail').textContent = user.email;
    $('#avatar').textContent = initials(user.name);
    $('#topAvatar').textContent = initials(user.name);
    $('#monthLabel').textContent = 'October 2026';
    await refresh();
}

async function refresh() {
    try {
        if (isDemoMode) {
            tx = (await api('/api/demo/transactions')).transactions;
            summary = await api('/api/demo/summary');
            forecast = await api('/api/demo/forecast');
        } else {
            tx = (await api('/api/transactions')).transactions;
            summary = await api('/api/summary');
            forecast = await api('/api/forecast');
        }
        try { mlStatus = await api('/api/ml/status'); } catch {}
        renderAll();
    } catch (err) {
        toast(err.message, true);
    }
}

function renderAll() {
    renderSummary();
    renderTx();
    renderExpenses();
    renderForecast();
    renderBudget();
    renderBills();
    renderGoals();
    renderSettings();
}

function renderSummary() {
    const s = summary;
    if (!s) return;
    $('#balance').textContent = money(s.balance);
    $('#income').textContent = money(s.income);
    $('#spent').textContent = money(s.spent);

    if (forecast && forecast.status === 'success') {
        $('#forecastCard').textContent = money(forecast.predicted_total);
        $('#forecastNote').textContent = forecast.overrun > 0 ? money(forecast.overrun) + ' above budget' : 'Within budget';
        $('#overrun').textContent = money(forecast.overrun);
        $('#warningText').textContent = forecast.overrun > 0
            ? `${forecast.highest_increase_category} has the highest projected increase.`
            : (isDemoMode ? 'Projected demo spending is within benchmark budget.' : 'Projected 30-day spending is within planned budget.');
    } else {
        $('#forecastCard').textContent = '—';
        $('#forecastNote').textContent = forecast ? (forecast.warning || 'Awaiting 30d history') : 'Calculating from history';
        $('#overrun').textContent = '₹0';
        $('#warningText').textContent = forecast?.warning || 'Add at least 30 days of transactions or explore Demo Mode to view forecast.';
    }

    $('#budgetSpent').textContent = money(s.spent);
    $('#budgetAmount').textContent = money(s.budget);
    $('#budgetRemaining').textContent = money(Math.max(0, s.budget - s.spent));

    let pct = s.budget ? Math.min(100, Math.round(s.spent / s.budget * 100)) : 0;
    $('#budgetPct').textContent = pct + '%';
    $('#budgetStatus').textContent = pct >= 90 ? 'Watch' : 'Healthy';
    $('#ring').style.background = `conic-gradient(var(--green) 0 ${pct * 3.6}deg,#e9edf1 ${pct * 3.6}deg)`;

    $('#donutTotal').textContent = money(s.spent);
    let cats = Object.entries(s.categories).sort((a, b) => b[1] - a[1]);
    $('#legend').innerHTML = cats.slice(0, 5).map(([k, v]) => `<div><span>${esc(k)}</span><b>${s.spent ? Math.round(v / s.spent * 100) : 0}%</b></div>`).join('') || '<div><span>No expenses yet</span><b>0%</b></div>';

    $('#recent').innerHTML = tx.slice(0, 8).map(row).join('');
    $('#miniList').innerHTML = tx.slice(0, 4).map(t => `<div class="mini-item"><div><b>${esc(t.merchant)}</b><small>${esc(t.category)} · ${fmtDate(t.date)}</small></div><b class="${t.type === 'Income' ? 'amount-income' : 'amount-expense'}">${t.type === 'Income' ? '+' : '−'}${money(t.amount)}</b></div>`).join('') || '<p>No transactions yet.</p>';
    drawChart();
}

function row(t) {
    return `<tr><td><b>${esc(t.merchant)}</b></td><td>${fmtDate(t.date)}</td><td>${esc(t.category)}</td><td><span class="type-tag">${esc(t.type)}</span></td><td class="${t.type === 'Income' ? 'amount-income' : 'amount-expense'}">${t.type === 'Income' ? '+' : '−'}${money(t.amount)}</td></tr>`;
}

function renderTx() {
    const q = ($('#search')?.value || '').toLowerCase();
    const f = $('#filter')?.value || 'All types';
    const a = tx.filter(t => (f === 'All types' || t.type === f) && (`${t.merchant} ${t.category}`.toLowerCase().includes(q)));
    $('#rows').innerHTML = a.map(t => `<tr><td><b>${esc(t.merchant)}</b></td><td>${fmtDate(t.date)}</td><td>${esc(t.category)}</td><td><span class="type-tag">${esc(t.type)}</span></td><td class="${t.type === 'Income' ? 'amount-income' : 'amount-expense'}">${t.type === 'Income' ? '+' : '−'}${money(t.amount)}</td><td>${isDemoMode ? '<span class="read-only-tag">Read-only</span>' : `<button class="secondary" onclick="removeTx(${t.id})">Delete</button>`}</td></tr>`).join('') || '<tr><td colspan="6">No matching transactions.</td></tr>';
}

window.removeTx = async id => {
    if (isDemoMode) {
        toast('Demo Mode is read-only. Deleting transactions is disabled.', true);
        return;
    }
    if (!confirm('Delete this transaction?')) return;
    try {
        await api('/api/transactions/' + id, { method: 'DELETE' });
        await refresh();
        toast('Transaction deleted');
    } catch (e) {
        toast(e.message, true);
    }
};

function renderExpenses() {
    if (!summary) return;
    const cats = Object.entries(summary.categories).sort((a, b) => b[1] - a[1]);
    $('#expTotal').textContent = money(summary.spent);
    $('#dailyAvg').textContent = money(summary.spent / Math.max(1, new Date().getDate()));
    $('#largestCat').textContent = cats[0]?.[0] || '—';
    $('#largestCatVal').textContent = money(cats[0]?.[1] || 0);
    const curMonthKey = isDemoMode && summary.reference_month ? summary.reference_month : new Date().toISOString().slice(0, 7);
    $('#recurring').textContent = money(tx.filter(t => t.recurring && t.type === 'Expense' && t.date.slice(0, 7) === curMonthKey).reduce((a, b) => a + b.amount, 0));
    const max = cats[0]?.[1] || 1;
    $('#bars').innerHTML = cats.map(([k, v]) => `<div class="bar-row"><span>${esc(k)}</span><div class="bar"><i style="width:${v / max * 100}%"></i></div><b>${money(v)}</b></div>`).join('') || '<p>No expense transactions this month.</p>';
}

function renderForecast() {
    if (!forecast) return;
    $('#engineText').textContent = isDemoMode
        ? 'Trained Sequential LSTM inference on Personal_Finance_Dataset.csv (30d window)'
        : (forecast.engine || 'Sequential LSTM (30d window)');
    $('#predBudget').textContent = money(forecast.budget);

    if (forecast.status === 'success') {
        $('#predicted').textContent = money(forecast.predicted_total);
        $('#predOverrun').textContent = forecast.overrun > 0 ? `+${money(forecast.overrun)}` : 'Within budget';
        $('#bigOverrun').textContent = money(forecast.overrun);
        $('#highCat').textContent = forecast.highest_increase_category || '—';
        $('#historyMonths').textContent = forecast.months_of_history || '0';
        $('#predDescription').textContent = forecast.overrun > 0
            ? `Projected 30-day spending is ${money(forecast.overrun)} above the planned budget.`
            : 'Projected 30-day spending is within the planned budget.';

        $('#predCats').innerHTML = Object.entries(forecast.category_predictions || {})
            .sort((a, b) => b[1] - a[1])
            .map(([k, v]) => `<div><span>${esc(k)}</span><b>${money(v)}</b></div>`)
            .join('') || '<p>No category breakdown available.</p>';
    } else {
        $('#predicted').textContent = '—';
        $('#predOverrun').textContent = '—';
        $('#bigOverrun').textContent = '—';
        $('#highCat').textContent = '—';
        $('#historyMonths').textContent = forecast.months_of_history || '0';
        $('#predDescription').textContent = forecast.warning || 'At least 30 days of transactions are required for genuine LSTM forecasting.';
        $('#predCats').innerHTML = `<p class="muted">${forecast.warning || 'Category predictions appear when you have recorded at least 30 days of expenses.'}</p>`;
    }
}

function renderBudget() {
    if (!summary) return;
    const pct = summary.budget ? Math.min(100, summary.spent / summary.budget * 100) : 0;
    $('#budgetInput').value = Math.round(summary.budget);
    if (isDemoMode) {
        $('#budgetInput').disabled = true;
    } else {
        $('#budgetInput').disabled = false;
    }
    $('#budgetBar').style.width = pct + '%';
    const predTxt = forecast && forecast.status === 'success' ? money(forecast.predicted_total) : 'Pending data';
    const note = isDemoMode ? ' · Benchmark Budget' : '';
    $('#budgetCopy').textContent = `Spent ${money(summary.spent)} · Remaining ${money(Math.max(0, summary.budget - summary.spent))} · Forecast ${predTxt}${note}`;
}

async function renderBills() {
    try {
        const d = isDemoMode ? await api('/api/demo/bills') : await api('/api/bills');
        const bills = d.bills || [];
        const container = $('#billsContainer');
        if (!container) return;

        if (bills.length === 0) {
            container.innerHTML = `
                <article class="panel">
                    <div class="empty">
                        <b>Bills are transaction-driven</b>
                        <p>Mark a transaction as recurring when you add it. Your recurring payments will remain stored with your account.</p>
                        <button class="primary" id="addBillInner">＋ Add recurring transaction</button>
                    </div>
                </article>`;
            const btn = $('#addBillInner');
            if (btn) btn.onclick = () => {
                if (isDemoMode) {
                    toast('Demo Mode is read-only. Modifying bills is disabled.', true);
                    return;
                }
                openModal();
                $('#recurringInput').checked = true;
            };
        } else {
            container.innerHTML = `
                <article class="panel">
                    <div class="panel-head">
                        <div>
                            <h2>Recurring bills & commitments</h2>
                            <p>${bills.length} active recurring payment(s) tracked${isDemoMode ? ' (Sample Data)' : ''}</p>
                        </div>
                    </div>
                    <div class="table-scroll">
                        <table>
                            <thead>
                                <tr><th>Bill / Payee</th><th>Last recorded date</th><th>Category</th><th>Amount</th><th></th></tr>
                            </thead>
                            <tbody>
                                ${bills.map(b => `
                                    <tr>
                                        <td><b>${esc(b.merchant)}</b></td>
                                        <td>${fmtDate(b.date)}</td>
                                        <td>${esc(b.category)}</td>
                                        <td class="amount-expense">−${money(b.amount)}</td>
                                        <td>${isDemoMode ? '<span class="read-only-tag">Sample</span>' : `<button class="secondary" onclick="removeTx(${b.id})">Delete</button>`}</td>
                                    </tr>`).join('')}
                            </tbody>
                        </table>
                    </div>
                </article>`;
        }
    } catch {}
}

async function renderGoals() {
    try {
        const d = isDemoMode ? await api('/api/demo/goals') : await api('/api/goals');
        $('#goalGrid').innerHTML = d.goals.map(g => {
            let p = Math.min(100, g.target ? g.current / g.target * 100 : 0);
            return `
            <article class="goal-card">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <h2>${esc(g.name)}</h2>
                    ${isDemoMode ? '<span class="read-only-tag">Demo</span>' : `<button class="secondary" style="padding:2px 8px;font-size:12px;" onclick="removeGoal(${g.id})">Delete</button>`}
                </div>
                <strong>${money(g.current)}</strong>
                <p>of ${money(g.target)}</p>
                <div class="progress"><i style="width:${p}%"></i></div>
                <small>${Math.round(p)}% complete</small>
            </article>`;
        }).join('') || '<article class="panel"><p>No goals yet. Add your first savings goal.</p></article>';
    } catch {}
}

window.removeGoal = async gid => {
    if (isDemoMode) {
        toast('Demo Mode is read-only. Deleting goals is disabled.', true);
        return;
    }
    if (!confirm('Delete this goal?')) return;
    try {
        await api('/api/goals/' + gid, { method: 'DELETE' });
        await renderGoals();
        toast('Goal deleted');
    } catch (e) {
        toast(e.message, true);
    }
};

function renderSettings() {
    if (!mlStatus) return;
    const statusElem = $('#settingsModelStatus');
    if (statusElem) {
        if (mlStatus.available) {
            statusElem.textContent = `Active · Test MAE: ${mlStatus.metrics?.mae || 'N/A'} · ${mlStatus.window_size_days}d window`;
        } else {
            statusElem.textContent = 'Model not loaded';
        }
    }
}

function drawChart() {
    const months = {};
    tx.filter(t => t.type === 'Expense').forEach(t => {
        months[t.date.slice(0, 7)] = (months[t.date.slice(0, 7)] || 0) + t.amount;
    });
    let keys = Object.keys(months).sort().slice(-6);
    if (chart) chart.destroy();
    chart = new Chart($('#chart'), {
        type: 'line',
        data: {
            labels: keys.length ? keys.map(x => x.slice(5) + ' / ' + x.slice(0, 4)) : ['No data'],
            datasets: [{
                label: 'Expenses',
                data: keys.map(k => months[k]),
                borderColor: '#1d8b72',
                backgroundColor: 'rgba(29,139,114,.08)',
                fill: true,
                tension: 0.35,
                pointRadius: 3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: '#eef1f4' },
                    ticks: { callback: v => '₹' + Number(v).toLocaleString('en-IN') }
                },
                x: { grid: { display: false } }
            }
        }
    });
}

function fmtDate(s) {
    const p = s.split('-');
    return p.length === 3 ? `${p[2]}-${p[1]}-${p[0]}` : s;
}

function esc(s) {
    return String(s).replace(/[&<>"']/g, m => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[m]));
}

function openModal() {
    if (isDemoMode) {
        toast('Demo Mode is read-only. Please log in or sign up to record new transactions.', true);
        return;
    }
    const d = new Date();
    $('#date').value = d.toISOString().slice(0, 10);
    $('#modal').classList.add('show');
}

function closeModal() {
    $('#modal').classList.remove('show');
}

$('#add').onclick = openModal;
$('#add2').onclick = openModal;
const addBillTop = $('#addBillTop');
if (addBillTop) addBillTop.onclick = () => {
    if (isDemoMode) {
        toast('Demo Mode is read-only. Adding bills is disabled.', true);
        return;
    }
    openModal();
    $('#recurringInput').checked = true;
};
$('#close').onclick = closeModal;
$('#modal').onclick = e => { if (e.target.id === 'modal') closeModal(); };

$('#form').onsubmit = async e => {
    e.preventDefault();
    if (isDemoMode) {
        toast('Demo Mode is read-only. Adding transactions is disabled.', true);
        closeModal();
        return;
    }
    try {
        await api('/api/transactions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                date: $('#date').value,
                merchant: $('#merchant').value,
                amount: Number($('#amount').value),
                type: $('#type').value,
                category: $('#cat').value,
                recurring: $('#recurringInput').checked
            })
        });
        closeModal();
        e.target.reset();
        await refresh();
        toast('Transaction saved');
    } catch (x) {
        toast(x.message, true);
    }
};

$('#search').oninput = renderTx;
$('#filter').onchange = renderTx;

const globalSearch = $('#globalSearch');
if (globalSearch) {
    globalSearch.oninput = () => {
        const val = globalSearch.value.trim();
        if ($('#search')) {
            $('#search').value = val;
            renderTx();
        }
    };
}

$('#csvInput').onchange = async e => {
    if (isDemoMode) {
        toast('Demo Mode is read-only. CSV import is disabled.', true);
        e.target.value = '';
        return;
    }
    const f = e.target.files[0];
    if (!f) return;
    try {
        const d = new FormData();
        d.append('file', f);
        const r = await api('/api/transactions/import', { method: 'POST', body: d });
        await refresh();
        toast(`${r.imported} transaction(s) imported${r.skipped ? `, ${r.skipped} skipped` : ''}`);
    } catch (x) {
        toast(x.message, true);
    }
    e.target.value = '';
};

$('#saveBudget').onclick = async () => {
    if (isDemoMode) {
        toast('Demo Mode uses a fixed benchmark budget (₹20,000). Modifying budget is disabled.', true);
        return;
    }
    try {
        const m = new Date().toISOString().slice(0, 7);
        await api('/api/budget', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ month: m, amount: Number($('#budgetInput').value) })
        });
        await refresh();
        toast('Budget updated');
    } catch (e) {
        toast(e.message, true);
    }
};

$('#addGoal').onclick = async () => {
    if (isDemoMode) {
        toast('Demo Mode is read-only. Adding goals is disabled.', true);
        return;
    }
    const name = prompt('Goal name');
    if (!name) return;
    const target = Number(prompt('Target amount (₹)') || 0);
    if (!target) return;
    try {
        await api('/api/goals', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, target, current: 0 })
        });
        await renderGoals();
        toast('Goal created');
    } catch (e) {
        toast(e.message, true);
    }
};

$$('nav button').forEach(b => b.onclick = () => go(b.dataset.page));
$$('[data-page]').forEach(b => {
    if (!b.closest('nav')) b.onclick = () => go(b.dataset.page);
});

function go(page) {
    $$('.page').forEach(x => x.classList.remove('active'));
    $('#' + page).classList.add('active');
    $$('nav button').forEach(x => x.classList.toggle('active', x.dataset.page === page));
    window.scrollTo(0, 0);
}

$('#menu').onclick = () => document.querySelector('aside').classList.toggle('open');

(async () => {
    try {
        user = (await api('/api/auth/me')).user;
        startApp();
    } catch {
        showAuth(false);
    }
})();
