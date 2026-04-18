// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewContainer = document.getElementById('previewContainer');
const previewImage = document.getElementById('previewImage');
const clearBtn = document.getElementById('clearBtn');
const predictBtn = document.getElementById('predictBtn');
const resultsSection = document.getElementById('resultsSection');
const predictedClass = document.getElementById('predictedClass');
const confidence = document.getElementById('confidence');
const probBars = document.getElementById('probBars');
const disposalGuide = document.getElementById('disposalGuide');
const loading = document.getElementById('loading');
const errorMessage = document.getElementById('errorMessage');

let selectedFile = null;

// Disposal guidelines for each waste type
const disposalGuidelines = {
    cardboard: {
        icon: '📦',
        title: 'Cardboard Disposal',
        tips: 'Flatten cardboard boxes before recycling. Remove any tape, labels, or plastic packaging. Keep cardboard dry - wet cardboard should go in compost or trash.'
    },
    glass: {
        icon: '🍶',
        title: 'Glass Disposal',
        tips: 'Rinse glass containers before recycling. Remove lids and caps (recycle separately if metal). Do not include broken glass, mirrors, or window glass in regular recycling.'
    },
    metal: {
        icon: '🥫',
        title: 'Metal Disposal',
        tips: 'Rinse cans and containers. Aluminum and steel cans are widely recyclable. Crush cans to save space. Keep metal lids attached or recycle separately.'
    },
    paper: {
        icon: '📄',
        title: 'Paper Disposal',
        tips: 'Keep paper clean and dry. Shredded paper can be recycled in paper bags. Remove plastic windows from envelopes. Avoid recycling paper with food residue.'
    },
    plastic: {
        icon: '♳',
        title: 'Plastic Disposal',
        tips: 'Check the recycling number (1-7) on the bottom. Rinse containers before recycling. Remove caps and labels when possible. Avoid plastic bags in regular recycling.'
    },
    trash: {
        icon: '🗑️',
        title: 'General Waste',
        tips: 'This item should go in general waste/landfill. Consider if any parts can be separated for recycling. Reduce consumption of non-recyclable items when possible.'
    }
};

// Event Listeners
uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

clearBtn.addEventListener('click', clearSelection);

predictBtn.addEventListener('click', classifyImage);

// Functions
function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        showError('Please select an image file.');
        return;
    }

    selectedFile = file;
    const reader = new FileReader();
    
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadArea.style.display = 'none';
        previewContainer.style.display = 'block';
        predictBtn.disabled = false;
        hideError();
        hideResults();
    };
    
    reader.readAsDataURL(file);
}

function clearSelection() {
    selectedFile = null;
    fileInput.value = '';
    previewImage.src = '';
    previewContainer.style.display = 'none';
    uploadArea.style.display = 'block';
    predictBtn.disabled = true;
    hideResults();
    hideError();
}

async function classifyImage() {
    if (!selectedFile) return;

    showLoading();
    hideError();
    hideResults();

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Classification failed');
        }

        displayResults(data);
    } catch (error) {
        showError(error.message);
    } finally {
        hideLoading();
    }
}

function displayResults(data) {
    // Display predicted class
    predictedClass.textContent = `${getClassIcon(data.predicted_class)} ${data.predicted_class}`;
    confidence.textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;

    // Display probability bars
    probBars.innerHTML = '';
    const sortedProbs = Object.entries(data.probabilities)
        .sort((a, b) => b[1] - a[1]);

    sortedProbs.forEach(([cls, prob]) => {
        const percentage = (prob * 100).toFixed(1);
        const item = document.createElement('div');
        item.className = 'prob-item';
        item.innerHTML = `
            <span class="prob-label">${cls}</span>
            <div class="prob-bar-container">
                <div class="prob-bar" style="width: ${Math.max(percentage, 5)}%">
                    ${percentage}%
                </div>
            </div>
        `;
        probBars.appendChild(item);
    });

    // Display disposal guide
    const guide = disposalGuidelines[data.predicted_class];
    disposalGuide.innerHTML = `
        <h4>${guide.icon} ${guide.title}</h4>
        <p>${guide.tips}</p>
    `;

    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function getClassIcon(cls) {
    const icons = {
        cardboard: '📦',
        glass: '🍶',
        metal: '🥫',
        paper: '📄',
        plastic: '♳',
        trash: '🗑️'
    };
    return icons[cls] || '♻️';
}

function showLoading() {
    loading.style.display = 'block';
    predictBtn.disabled = true;
}

function hideLoading() {
    loading.style.display = 'none';
    predictBtn.disabled = false;
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
}

function hideError() {
    errorMessage.style.display = 'none';
}

function hideResults() {
    resultsSection.style.display = 'none';
}
