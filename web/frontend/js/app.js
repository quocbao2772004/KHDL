const API_BASE = 'https://distinct-adapted-hippo.ngrok-free.app';
console.log("MovieFlix App.js Loaded - Version 22 - " + new Date().toLocaleString());
let currentUser = null;
let heroMovie = null;

// --- Auth Logic ---
async function checkAuth() {
    console.log("Checking auth...");
    const storedUser = localStorage.getItem('user');
    if (storedUser) {
        try {
            currentUser = JSON.parse(storedUser);
            console.log("User logged in:", currentUser.username);
            updateAuthUI(true);
            await fetchUserRatings(); // Fetch ratings on login
            initRouter();
        } catch (e) {
            console.error("Error parsing user:", e);
            logout();
        }
    } else {
        console.log("No user found.");
        updateAuthUI(false);
        // If on page.html and not logged in, redirect to index
        if (window.location.pathname.includes('page.html')) {
            console.log("Redirecting to index.html (not logged in)");
            window.location.href = 'index.html';
        }
    }
}

function updateAuthUI(isLoggedIn) {
    const authOverlay = document.getElementById('authOverlay');
    const app = document.getElementById('app');
    const userDisplay = document.getElementById('userDisplay');

    if (isLoggedIn) {
        if (authOverlay) authOverlay.style.display = 'none';
        if (app) app.style.display = 'block';
        if (userDisplay) userDisplay.textContent = currentUser.username;
    } else {
        if (authOverlay) authOverlay.style.display = 'flex';
        if (app) app.style.display = 'none';
    }
}

function showRegister() {
    document.getElementById('loginBox').style.display = 'none';
    document.getElementById('registerBox').style.display = 'block';
}

function showLogin() {
    document.getElementById('registerBox').style.display = 'none';
    document.getElementById('loginBox').style.display = 'block';
}

async function handleLogin() {
    const user = document.getElementById('loginUser').value;
    const pass = document.getElementById('loginPass').value;

    try {
        const res = await fetch(`${API_BASE}/login`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'ngrok-skip-browser-warning': 'true'
            },
            body: JSON.stringify({ username: user, password: pass })
        });

        if (res.ok) {
            const data = await res.json();
            currentUser = data;
            localStorage.setItem('user', JSON.stringify(data));
            await fetchUserRatings(); // Fetch ratings on login
            checkAuth();
        } else {
            alert('Invalid credentials');
        }
    } catch (e) {
        console.error(e);
        alert('Login failed: ' + e.message);
    }
}

async function handleRegister() {
    const user = document.getElementById('regUser').value;
    const pass = document.getElementById('regPass').value;

    try {
        const res = await fetch(`${API_BASE}/register`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'ngrok-skip-browser-warning': 'true'
            },
            body: JSON.stringify({ username: user, password: pass })
        });

        if (res.ok) {
            alert('Registration successful! Please login.');
            showLogin();
        } else {
            alert('Registration failed (Username might exist)');
        }
    } catch (e) {
        console.error(e);
    }
}

function logout() {
    localStorage.removeItem('user');
    window.location.href = 'index.html';
}

// --- Data Fetching ---
// --- Data Fetching ---
async function fetchMovies(page = 1, limit = 50) {
    try {
        const res = await fetch(`${API_BASE}/movies?page=${page}&limit=${limit}`, {
            headers: { 'ngrok-skip-browser-warning': 'true' }
        });
        return await res.json();
    } catch (e) {
        console.error(e);
        return { movies: [], total: 0, page: 1, limit: limit };
    }
}

// --- Routing & Initialization ---
function initRouter() {
    const path = window.location.pathname;
    const page = path.split('/').pop();
    console.log("Current page:", page);

    if (path.includes('page.html')) {
        initMoviePage();
    } else if (path.includes('index.html') || page === '' || page === '/') {
        initHomePage();
    } else {
        // Fallback for other paths if necessary
        console.log("Unknown path, defaulting to home init if app exists");
        if (document.getElementById('trendingRow')) {
            initHomePage();
        }
    }
}

let homeInitialized = false;
async function initHomePage() {
    if (homeInitialized) return;
    homeInitialized = true;
    console.log("Initializing Home Page");

    // Navbar scroll effect
    window.addEventListener('scroll', () => {
        const nav = document.getElementById('navbar');
        if (nav) {
            if (window.scrollY > 100) nav.classList.add('scrolled');
            else nav.classList.remove('scrolled');
        }
    });

    // Initial load
    const params = new URLSearchParams(window.location.search);
    const view = params.get('view');
    const searchQuery = params.get('search');

    if (view === 'evaluation') {
        showEvaluation();
    } else if (view === 'analysis') {
        showAnalysis();
    } else if (searchQuery) {
        console.log("Search query found on init:", searchQuery);
        const searchInput = document.getElementById('searchInput');
        if (searchInput) searchInput.value = searchQuery;
        await performSearch(searchQuery);
    } else {
        // Initial Load (only if no search)
        await loadAllMovies(1);
        // Trending Row
        const trendingData = await fetchMovies(1, 20);
        renderGrid('trendingRow', trendingData.movies || []);
        showHome();
    }

    // Manual fallback for nav links if onclick fails
    const navLinks = document.querySelectorAll('.nav-links span');
    navLinks.forEach(span => {
        span.addEventListener('click', (e) => {
            const text = e.target.innerText.trim();
            if (text === 'Home') showHome();
            else if (text === 'Evaluation') showEvaluation();
            else if (text === 'Analysis') showAnalysis();
        });
    });
}

function hideAllViews() {
    const views = ['home-view', 'movie-view', 'evaluation-view', 'analysis-view', 'searchResults'];
    views.forEach(v => {
        const el = document.getElementById(v);
        if (el) el.style.display = 'none';
    });
}

function showHome() {
    console.log("Showing Home View");
    if (!document.getElementById('home-view')) {
        window.location.href = 'index.html';
        return;
    }
    hideAllViews();
    const homeView = document.getElementById('home-view');
    if (homeView) homeView.style.display = 'block';
    window.scrollTo(0, 0);
}

function showEvaluation() {
    console.log("Showing Evaluation View");
    if (!document.getElementById('evaluation-view')) {
        window.location.href = 'index.html?view=evaluation';
        return;
    }
    hideAllViews();
    const evalView = document.getElementById('evaluation-view');
    if (evalView) {
        evalView.style.display = 'block';
        window.scrollTo(0, 0);
        renderEvaluationCharts();
    }
}

function showAnalysis() {
    console.log("Showing Analysis View");
    if (!document.getElementById('analysis-view')) {
        window.location.href = 'index.html?view=analysis';
        return;
    }
    hideAllViews();
    const analysisView = document.getElementById('analysis-view');
    if (analysisView) {
        analysisView.style.display = 'block';
        window.scrollTo(0, 0);
    }
}

function renderEvaluationCharts() {
    if (typeof Chart === 'undefined') {
        console.warn("Chart.js not loaded yet. Retrying in 500ms...");
        setTimeout(renderEvaluationCharts, 500);
        return;
    }
    // BERT Chart
    const bertCanvas = document.getElementById('bertChart');
    if (bertCanvas) {
        const bertCtx = bertCanvas.getContext('2d');
        new Chart(bertCtx, {
            type: 'bar',
            data: {
                labels: ['Precision@10', 'Recall@10'],
                datasets: [{
                    label: 'BERT Performance',
                    data: [0.7740, 0.7740],
                    backgroundColor: ['rgba(229, 9, 20, 0.7)', 'rgba(229, 9, 20, 0.4)'],
                    borderColor: ['rgba(229, 9, 20, 1)', 'rgba(229, 9, 20, 1)'],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: { beginAtZero: true, max: 1, grid: { color: '#333' }, ticks: { color: '#ccc' } },
                    x: { ticks: { color: '#ccc' } }
                },
                plugins: { legend: { labels: { color: '#fff' } } }
            }
        });
    }

    // LightGCN Chart
    const gcnCanvas = document.getElementById('lightgcnChart');
    if (gcnCanvas) {
        const gcnCtx = gcnCanvas.getContext('2d');
        new Chart(gcnCtx, {
            type: 'bar',
            data: {
                labels: ['Recall@5', 'Recall@10', 'Recall@20', 'Precision@5', 'Precision@10', 'Precision@20'],
                datasets: [{
                    label: 'LightGCN Metrics (Test Set)',
                    data: [0.9622, 0.9934, 0.9995, 0.3625, 0.2009, 0.1036],
                    backgroundColor: 'rgba(70, 211, 105, 0.6)',
                    borderColor: 'rgba(70, 211, 105, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: { beginAtZero: true, max: 1.1, grid: { color: '#333' }, ticks: { color: '#ccc' } },
                    x: { ticks: { color: '#ccc' } }
                },
                plugins: { legend: { labels: { color: '#fff' } } }
            }
        });
    }
}


let currentPage = 1;
const itemsPerPage = 50;

async function loadAllMovies(page) {
    currentPage = page;
    const data = await fetchMovies(page, itemsPerPage);

    if (data && data.movies) {
        if (page === 1 && !heroMovie) {
            heroMovie = data.movies[Math.floor(Math.random() * data.movies.length)];
            setHero(heroMovie);
        }

        renderGrid('allMoviesGrid', data.movies);
        renderPagination(data.total, data.page, data.limit);
    }

    // Personalized Recommendations (if logged in) - only on first load
    if (page === 1 && currentUser) {
        await fetchAndRenderUserRecs();
    }
}

function renderPagination(total, page, limit) {
    const container = document.getElementById('paginationControls');
    if (!container) return;

    const totalPages = Math.ceil(total / limit);

    container.innerHTML = `
        <button onclick="changePage(${page - 1})" ${page <= 1 ? 'disabled' : ''} style="padding: 10px 20px; cursor: pointer; background: #333; color: white; border: none; border-radius: 4px;">Previous</button>
        <span style="align-self: center; font-size: 1.1rem;">Page ${page} of ${totalPages}</span>
        <button onclick="changePage(${page + 1})" ${page >= totalPages ? 'disabled' : ''} style="padding: 10px 20px; cursor: pointer; background: #333; color: white; border: none; border-radius: 4px;">Next</button>
    `;
}

async function changePage(newPage) {
    if (newPage < 1) return;
    await loadAllMovies(newPage);
    document.getElementById('allMoviesSection').scrollIntoView({ behavior: 'smooth' });
}

let movieInitialized = false;
async function initMoviePage() {
    if (movieInitialized) return;
    movieInitialized = true;
    console.log("Initializing Movie Page");

    const params = new URLSearchParams(window.location.search);
    const title = params.get('title');
    console.log("Movie title:", title);

    if (!title) {
        console.log("No title found, redirecting to index");
        window.location.href = 'index.html';
        return;
    }

    try {
        // Fetch Details
        console.log("Fetching details for:", title);
        const res = await fetch(`${API_BASE}/movie/${encodeURIComponent(title)}`, {
            headers: { 'ngrok-skip-browser-warning': 'true' }
        });
        if (res.ok) {
            const movie = await res.json();
            renderMovieDetails(movie);

            // Log Action
            logAction('view_details', `Viewed: ${movie.title}`);

            // Fetch Recommendations
            console.log("Fetching recommendations for:", title);
            const recRes = await fetch(`${API_BASE}/recommend/${encodeURIComponent(title)}?top_k=30`, {
                headers: { 'ngrok-skip-browser-warning': 'true' }
            });
            if (recRes.ok) {
                const recs = await recRes.json();
                renderGrid('detailsRecs', recs);
            }

            // Fetch CF recommendations if user is logged in
            if (currentUser && movie.tmdb_id) {
                fetchCFRecommendations(movie.tmdb_id);
            }
        } else {
            console.error("Movie not found API error");
            alert('Movie not found');
            window.location.href = 'index.html';
        }
    } catch (e) {
        console.error("Error in initMoviePage:", e);
    }
}

// --- Rendering Helpers ---

function setHero(movie) {
    const titleEl = document.getElementById('heroTitle');
    const descEl = document.getElementById('heroOverview');
    const heroEl = document.getElementById('hero');

    if (titleEl) titleEl.textContent = movie.title;
    if (descEl) descEl.textContent = truncate(movie.overview, 150);

    const poster = movie.poster_path || '';
    if (heroEl) heroEl.style.backgroundImage = `url('${poster}')`;
}

function renderMovieDetails(movie) {
    const titleEl = document.getElementById('detailsTitle');
    const overviewEl = document.getElementById('detailsOverview');
    const metaEl = document.getElementById('detailsMeta');
    const heroEl = document.getElementById('detailsHero');

    if (titleEl) titleEl.textContent = movie.title;
    if (overviewEl) overviewEl.textContent = movie.overview;
    if (metaEl) {
        // Parse genres safely
        let genresHtml = '';
        try {
            let genres = movie.genres;
            if (typeof genres === 'string') {
                // Handle python list string representation
                genres = genres.replace(/'/g, '"');
                genres = JSON.parse(genres);
            }
            if (Array.isArray(genres)) {
                genresHtml = `<div class="genre-list">
                    ${genres.map(g => `<span class="genre-badge">${g}</span>`).join('')}
                </div>`;
            }
        } catch (e) {
            console.warn("Error parsing genres:", e);
            genresHtml = movie.genres || '';
        }

        const date = movie.release_date || 'Unknown Date';
        const rating = movie.score ? `★ ${movie.score.toFixed(1)}` : 'N/A';

        // Parse cast
        let castHtml = '';
        try {
            let cast = movie.cast;
            if (typeof cast === 'string') {
                cast = cast.replace(/'/g, '"');
                cast = JSON.parse(cast);
            }
            if (Array.isArray(cast)) {
                const formattedCast = cast.slice(0, 5).map(name => {
                    return name.split('_')
                        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                        .join(' ');
                });
                castHtml = `<div class="meta-item" title="Main Actor">
                    🎭 Main Actor: ${formattedCast.join(', ')}
                </div>`;
            }
        } catch (e) {
            console.warn("Error parsing cast:", e);
        }

        let directorHtml = '';
        if (movie.director) {
            const formattedDirector = movie.director.split('_')
                .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                .join(' ');
            directorHtml = `<div class="meta-item" title="Director">🎬 Director: ${formattedDirector}</div>`;
        }

        metaEl.innerHTML = `
            <div class="meta-container">
                <div class="meta-item" title="Release Date">
                    📅 ${date}
                </div>
                <div class="meta-item" style="color: #ffd700;" title="Rating">
                    ${rating}
                </div>
                ${directorHtml}
                ${castHtml}
                ${genresHtml}
            </div>
        `;

        // Add Rating Section
        const ratingContainer = document.createElement('div');
        ratingContainer.id = 'userRatingContainer';
        ratingContainer.style.marginTop = '10px';
        metaEl.appendChild(ratingContainer);

        // Render initial rating state
        const myRating = userRatings[movie.tmdb_id] || 0;
        console.log(`Initial rating for movie ${movie.tmdb_id}:`, myRating);
        renderRatingUI(movie.tmdb_id, myRating);

    }

    const poster = movie.poster_path || '';
    if (heroEl) heroEl.style.backgroundImage = `url('${poster}')`;
}

function renderRow(elementId, movies) {
    const container = document.getElementById(elementId);
    if (!container) return;
    container.innerHTML = '';
    movies.forEach(m => {
        const div = document.createElement('div');
        div.className = 'poster-container';
        div.style.position = 'relative';
        div.style.display = 'inline-block';
        div.style.marginRight = '10px';
        div.style.cursor = 'pointer';
        div.onclick = () => goToMovie(m.title);

        const img = document.createElement('img');
        img.className = 'poster';
        img.src = m.poster_path || 'https://via.placeholder.com/150x225';
        img.style.display = 'block';

        const scoreHtml = m.score ? `<div style="position:absolute; top:5px; right:5px; background:rgba(0,0,0,0.7); color:#46d369; padding:2px 6px; border-radius:4px; font-size:0.8rem; font-weight:bold;">${(m.score * 100).toFixed(0)}%</div>` : '';

        div.innerHTML = `
            ${img.outerHTML}
            ${scoreHtml}
        `;

        div.onmouseenter = () => div.querySelector('img').style.transform = 'scale(1.05)';
        div.onmouseleave = () => div.querySelector('img').style.transform = 'scale(1)';

        container.appendChild(div);
    });
}

function renderGrid(elementId, movies) {
    const container = document.getElementById(elementId);
    if (!container) return;
    container.innerHTML = '';
    movies.forEach(m => {
        const div = document.createElement('div');
        div.style.cursor = 'pointer';
        div.style.position = 'relative';
        div.onclick = () => goToMovie(m.title);

        let scoreHtml = '';
        if (m.score !== null && m.score !== undefined) {
            const scoreVal = parseFloat(m.score);
            if (!isNaN(scoreVal)) {
                scoreHtml = `<div style="position:absolute; top:8px; right:8px; background:rgba(0,0,0,0.8); backdrop-filter: blur(4px); color:#46d369; padding:4px 8px; border-radius:6px; font-size:0.85rem; font-weight:700; box-shadow: 0 2px 4px rgba(0,0,0,0.3); z-index: 2;">${(scoreVal * 100).toFixed(0)}%</div>`;
            }
        }

        // Check for user rating (Below Score)
        let userRatingHtml = '';
        if (currentUser && m.tmdb_id && userRatings[m.tmdb_id]) {
            // Calculate top position based on whether score exists
            const topPos = m.score ? '40px' : '8px';
            userRatingHtml = `
                <div style="position:absolute; top:${topPos}; right:8px; background:rgba(0,0,0,0.8); backdrop-filter: blur(4px); color:#ffd700; padding:4px 8px; border-radius:6px; font-size:0.85rem; font-weight:600; display:flex; align-items:center; gap:4px; box-shadow: 0 2px 4px rgba(0,0,0,0.3); z-index: 2;">
                    <span style="font-size: 1rem;">★</span> ${userRatings[m.tmdb_id]}
                </div>`;
        }

        // Check for average rating (Bottom Right)
        let avgRatingHtml = '';
        if (m.average_rating) {
            avgRatingHtml = `
                <div style="position:absolute; bottom:8px; right:8px; background:rgba(0,0,0,0.8); backdrop-filter: blur(4px); color:#fff; padding:4px 8px; border-radius:6px; font-size:0.85rem; font-weight:600; display:flex; align-items:center; gap:4px; box-shadow: 0 2px 4px rgba(0,0,0,0.3); z-index: 2;">
                    <span style="color:#ffd700; font-size: 1rem;">★</span> ${m.average_rating.toFixed(1)}
                </div>`;
        }

        div.innerHTML = `
            <div style="position:relative; overflow: hidden; border-radius: 8px;">
                <img src="${m.poster_path || 'https://via.placeholder.com/150x225'}" style="width:100%; display: block; transition: transform 0.3s ease;">
                ${scoreHtml}
                ${userRatingHtml}
                ${avgRatingHtml}
            </div>
            <div style="font-size:0.95rem; margin-top:8px; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: #e0e0e0;">${m.title}</div>
        `;
        div.onmouseenter = () => div.querySelector('img').style.transform = 'scale(1.05)';
        div.onmouseleave = () => div.querySelector('img').style.transform = 'scale(1)';

        container.appendChild(div);
    });
}

function goToMovie(title) {
    console.log("Navigating to movie:", title);
    window.location.href = `page.html?title=${encodeURIComponent(title)}`;
}

async function triggerSearch() {
    const input = document.getElementById('searchInput');
    if (!input) return;
    const query = input.value.toLowerCase();
    await performSearch(query);
}

async function handleSearch(e) {
    if (e.key === 'Enter') {
        const query = e.target.value.toLowerCase();
        await performSearch(query);
    }
}

function handleSearchInput(e) {
    const query = e.target.value;
    if (!query || query.trim() === '') {
        showHome();
    }
}


async function performSearch(query) {
    await logAction('search', `Searched for: ${query}`);

    // Update recommendations in background
    if (currentUser) {
        fetchAndRenderUserRecs();
    }

    if (window.location.pathname.includes('page.html')) {
        window.location.href = `index.html?search=${encodeURIComponent(query)}`;
        return;
    }

    try {
        const res = await fetch(`${API_BASE}/search?q=${encodeURIComponent(query)}`, {
            headers: { 'ngrok-skip-browser-warning': 'true' }
        });
        if (res.ok) {
            const results = await res.json();
            const searchContainer = document.getElementById('searchResults');
            if (searchContainer) {
                hideAllViews();
                searchContainer.style.display = 'block';
                renderGrid('searchGrid', results);
            }
        }
    } catch (e) {
        console.error("Search error:", e);
    }
}

function playHero() {
    if (heroMovie) goToMovie(heroMovie.title);
}

function infoHero() {
    if (heroMovie) goToMovie(heroMovie.title);
}

// --- Logging ---
async function logAction(action, details) {
    if (!currentUser) return;
    try {
        await fetch(`${API_BASE}/log`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'ngrok-skip-browser-warning': 'true'
            },
            body: JSON.stringify({
                user_id: currentUser.user_id,
                action: action,
                details: details
            })
        });
    } catch (e) {
        console.error("Log failed", e);
    }
}

// --- CF Recommendations ---

function displayCFRecommendations(recs, tmdb_id) {
    if (!recs || recs.length === 0) {
        console.warn(`No CF recommendations to display for movie ${tmdb_id}. API returned empty list.`);
        const cfSection = document.getElementById('cfRecsSection');
        if (cfSection) {
            cfSection.style.display = 'none';
        }
        return;
    }

    console.log(`Displaying ${recs.length} CF recommendations`);

    let cfSection = document.getElementById('cfRecsSection');
    if (!cfSection) {
        const detailsRecs = document.getElementById('detailsRecs');
        // If detailsRecs doesn't exist (maybe we are not on movie page), try to find a place
        // But usually this is for movie detail page.
        if (detailsRecs && detailsRecs.parentElement) {
            cfSection = document.createElement('div');
            cfSection.className = 'row';
            cfSection.id = 'cfRecsSection';
            cfSection.style.marginTop = '40px';
            cfSection.innerHTML = `
                <h3>You may also like</h3>
                <div class="rec-grid" id="cfRecs"></div>
            `;
            detailsRecs.parentElement.appendChild(cfSection);
        } else {
            console.warn("Cannot find detailsRecs container to append CF section");
            return;
        }
    }

    cfSection.style.display = 'block';
    renderGrid('cfRecs', recs);
    console.log("CF recommendations displayed");
}

async function fetchCFRecommendations(tmdb_id) {
    if (!currentUser || !tmdb_id) {
        console.log("Cannot fetch CF recommendations: missing user or tmdb_id");
        return;
    }

    try {
        const userRating = userRatings[tmdb_id] || null;
        let url = `${API_BASE}/recommend/movie/${tmdb_id}?top_k=30`;
        if (userRating !== null && userRating !== undefined) {
            url += `&user_rating=${userRating}&user_id=${currentUser.user_id}`;
        }

        console.log(`Fetching CF recommendations for movie ${tmdb_id}...`);
        const res = await fetch(url, {
            headers: { 'ngrok-skip-browser-warning': 'true' }
        });
        if (res.ok) {
            const recs = await res.json();
            console.log(`Received ${recs ? recs.length : 0} CF recommendations for movie ${tmdb_id}`);
            if (recs && recs.length > 0) {
                displayCFRecommendations(recs, tmdb_id);
            } else {
                const cfSection = document.getElementById('cfRecsSection');
                if (cfSection) {
                    cfSection.style.display = 'none';
                }
            }
        } else {
            console.error("Failed to fetch CF recommendations:", res.status, res.statusText);
        }
    } catch (e) {
        console.error("Error fetching CF recommendations:", e);
    }
}

// --- Rating Logic ---
async function rateMovie(movieId, rating) {
    if (!currentUser) {
        alert("Please login to rate movies.");
        return;
    }
    try {
        const res = await fetch(`${API_BASE}/rate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'ngrok-skip-browser-warning': 'true'
            },
            body: JSON.stringify({
                user_id: currentUser.user_id,
                movie_id: movieId,
                rating: rating
            })
        });
        if (res.ok) {
            const data = await res.json();
            console.log(`Rated movie ${movieId} with ${rating}`);
            // Update UI to reflect new rating
            renderRatingUI(movieId, rating);
            // Update userRatings cache
            userRatings[movieId] = rating;

            // Display recommendations from response
            if (data.recommendations && data.recommendations.length > 0) {
                displayCFRecommendations(data.recommendations, movieId);
            } else {
                // Fallback: Fetch CF recommendations
                fetchCFRecommendations(movieId);
            }

            // Also refresh home page recommendations if we are there or go there
            fetchAndRenderUserRecs();
        } else {
            alert("Failed to save rating.");
        }
    } catch (e) {
        console.error("Error rating movie:", e);
    }
}

let userRatings = {};
async function fetchUserRatings() {
    if (!currentUser) return;
    try {
        const res = await fetch(`${API_BASE}/ratings/${currentUser.user_id}`, {
            headers: { 'ngrok-skip-browser-warning': 'true' }
        });
        if (res.ok) {
            const data = await res.json();
            userRatings = {};
            data.forEach(r => {
                userRatings[r.tmdb_id] = r.rating;
            });
            console.log("User ratings loaded:", Object.keys(userRatings).length);
        }
    } catch (e) {
        console.error("Error fetching ratings:", e);
    }
}

function renderRatingUI(movieId, currentRating) {
    const container = document.getElementById('userRatingContainer');
    if (!container) return;

    let starsHtml = '';
    for (let i = 1; i <= 5; i++) {
        const filled = i <= currentRating ? '★' : '☆';
        const color = i <= currentRating ? '#ffd700' : '#ccc';
        starsHtml += `<span style="cursor: pointer; color: ${color}; font-size: 1.5rem;" onclick="rateMovie(${movieId}, ${i})">${filled}</span>`;
    }
    container.innerHTML = `Your Rating: ${starsHtml}`;
}


function truncate(str, n) {
    if (!str) return '';
    return (str.length > n) ? str.substr(0, n - 1) + '&hellip;' : str;
}

// Start
// --- New Helper for Recs ---
async function fetchAndRenderUserRecs() {
    if (!currentUser) return;
    console.log("Fetching recommendations for user:", currentUser.user_id);
    try {
        // Add timestamp to prevent caching
        const recRes = await fetch(`${API_BASE}/recommend/user/${currentUser.user_id}?_=${new Date().getTime()}`, {
            headers: { 'ngrok-skip-browser-warning': 'true' }
        });
        if (recRes.ok) {
            const recs = await recRes.json();
            console.log("Recs received:", recs.length);
            if (recs.length > 0) {
                // Create or update recommendation row
                let recContainer = document.getElementById('recRow');
                if (!recContainer) {
                    const mainContent = document.getElementById('mainContent');
                    const allMoviesSection = document.getElementById('allMoviesSection');

                    // Check if mainContent exists (it might not if we are on page.html, but this function is mostly for index.html)
                    if (!mainContent) return;

                    const recSection = document.createElement('div');
                    recSection.id = 'recSection';
                    recSection.style.marginTop = '60px';
                    recSection.innerHTML = `<h3>Recommended for You</h3><div class="rec-grid" id="recRow"></div>`;

                    // Insert before All Movies (so it's after Trending)
                    if (allMoviesSection) {
                        mainContent.insertBefore(recSection, allMoviesSection);
                    } else {
                        mainContent.appendChild(recSection);
                    }
                    recContainer = document.getElementById('recRow');
                }
                renderGrid('recRow', recs);
            }
        }
    } catch (e) {
        console.error("Error fetching user recommendations:", e);
    }
}

// Handle back navigation (bfcache)
window.addEventListener('pageshow', (event) => {
    if (event.persisted || (window.performance && window.performance.navigation.type === 2)) {
        console.log('Page restored, refreshing recs...');
        if (currentUser && window.location.pathname.includes('index.html')) {
            fetchAndRenderUserRecs();
        }
    }
});

// Start
checkAuth();