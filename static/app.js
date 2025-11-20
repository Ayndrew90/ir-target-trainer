const img = document.getElementById('video-stream');
const canvas = document.getElementById('roi-canvas');
const ctx = canvas.getContext('2d');
const roiDisplay = document.getElementById('roi-display');
const scoreDisplay = document.getElementById('score-display');
const statusMessage = document.getElementById('status-message');
const clearRoiButton = document.getElementById('clear-roi');
const clearHitsButton = document.getElementById('clear-hits');
const videoWrapper = document.getElementById('video-wrapper');

const BASE_WIDTH = parseInt(videoWrapper.dataset.baseWidth, 10);
const BASE_HEIGHT = parseInt(videoWrapper.dataset.baseHeight, 10);

let firstCorner = null;
let currentRoiDisplay = null;
let roiLocked = false;

function syncCanvasSize() {
    const rect = img.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
    drawOverlay();
}

function clearOverlayOnly() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function drawOverlay() {
    clearOverlayOnly();
    if (roiLocked) {
        return;
    }
    if (currentRoiDisplay) {
        const { x, y, w, h } = currentRoiDisplay;
        ctx.strokeStyle = 'lime';
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, w, h);
    }
    if (firstCorner) {
        ctx.fillStyle = 'yellow';
        ctx.beginPath();
        ctx.arc(firstCorner.x, firstCorner.y, 4, 0, Math.PI * 2);
        ctx.fill();
    }
}

canvas.addEventListener('click', (e) => {
    if (roiLocked) {
        return;
    }

    const rect = canvas.getBoundingClientRect();
    const xDisplay = e.clientX - rect.left;
    const yDisplay = e.clientY - rect.top;

    if (!firstCorner) {
        firstCorner = { x: xDisplay, y: yDisplay };
        statusMessage.textContent = "First corner set. Tap second corner (bottom-right).";
        drawOverlay();
    } else {
        const second = { x: xDisplay, y: yDisplay };

        let x1 = Math.min(firstCorner.x, second.x);
        let y1 = Math.min(firstCorner.y, second.y);
        let x2 = Math.max(firstCorner.x, second.x);
        let y2 = Math.max(firstCorner.y, second.y);

        let wDisplay = x2 - x1;
        let hDisplay = y2 - y1;

        if (wDisplay < 5 || hDisplay < 5) {
            statusMessage.textContent = "ROI too small, not set.";
            firstCorner = null;
            currentRoiDisplay = null;
            drawOverlay();
            return;
        }

        currentRoiDisplay = { x: x1, y: y1, w: wDisplay, h: hDisplay };
        drawOverlay();

        const scaleX = BASE_WIDTH / canvas.clientWidth;
        const scaleY = BASE_HEIGHT / canvas.clientHeight;

        const xBase = Math.round(x1 * scaleX);
        const yBase = Math.round(y1 * scaleY);
        const wBase = Math.round(wDisplay * scaleX);
        const hBase = Math.round(hDisplay * scaleY);

        fetch('/set_roi', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ x: xBase, y: yBase, w: wBase, h: hBase })
        }).then(response => response.json())
          .then(data => {
              roiDisplay.textContent = "ROI: x=" + data.x + ", y=" + data.y + ", w=" + data.w + ", h=" + data.h;
              statusMessage.textContent = "ROI set. Stream is now cropped.";
              roiLocked = true;
              currentRoiDisplay = null;
              firstCorner = null;
              drawOverlay();
          }).catch(err => {
              console.error(err);
              statusMessage.textContent = "Error setting ROI.";
              firstCorner = null;
              currentRoiDisplay = null;
              drawOverlay();
          });
    }
});

clearRoiButton.addEventListener('click', () => {
    fetch('/clear_roi', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            currentRoiDisplay = null;
            firstCorner = null;
            roiLocked = false;
            drawOverlay();
            roiDisplay.textContent = "ROI: (not set)";
            statusMessage.textContent = "ROI cleared. Full image shown.";
        })
        .catch(err => {
            console.error(err);
            statusMessage.textContent = "Error clearing ROI.";
        });
});

clearHitsButton.addEventListener('click', () => {
    fetch('/clear_hits', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            statusMessage.textContent = "All hits cleared.";
            updateStats();
        })
        .catch(err => {
            console.error(err);
            statusMessage.textContent = "Error clearing hits.";
        });
});

function updateStats() {
    fetch('/stats')
        .then(response => response.json())
        .then(data => {
            const score = data.score || 0;
            const avg = data.average || 0;
            const avgStr = avg.toFixed(1).replace('.', ',');
            scoreDisplay.textContent = "Score: " + score + " (Ø " + avgStr + ")";
        })
        .catch(err => {
            console.error(err);
        });
}

img.addEventListener('load', () => {
    syncCanvasSize();
    updateStats();
});

window.addEventListener('resize', syncCanvasSize);
setInterval(updateStats, 1000);
