// Esperar a que todo el DOM esté cargado
document.addEventListener('DOMContentLoaded', () => {

    // --- 1. SELECCIÓN DE ELEMENTOS DEL DOM ---
    const form = document.getElementById('prediction-form');
    if (!form) return; // Si no estamos en la página del modelo, no hacer nada

    const hspSlider = document.getElementById('hsp');
    const hspValue = document.getElementById('hsp-value');
    const healingTimeSlider = document.getElementById('healingTime');
    const healingTimeValue = document.getElementById('healingTime-value');
    
    const predictButton = document.getElementById('predict-button');
    const buttonText = document.getElementById('button-text');
    const buttonLoader = document.getElementById('button-loader');
    
    const resultsPlaceholder = document.getElementById('results-placeholder');
    const resultsContainer = document.getElementById('results-container');
    
    // Elementos de resultados
    const resultUtsEfficiency = document.getElementById('result-uts-efficiency');
    const resultElongationEfficiency = document.getElementById('result-elongation-efficiency');
    
    // Elementos del QR
    const qrCodeImg = document.getElementById('qr-code-img');
    const qrPlaceholder = document.getElementById('qr-placeholder');
    const downloadQrBtn = document.getElementById('download-qr-btn');

    // --- 2. ACTUALIZACIÓN DE VALORES DE SLIDERS ---
    hspSlider.addEventListener('input', (e) => {
        hspValue.textContent = parseFloat(e.target.value).toFixed(2);
    });
    
    healingTimeSlider.addEventListener('input', (e) => {
        healingTimeValue.textContent = `${parseFloat(e.target.value).toFixed(1)} h`;
    });

    // --- 3. MANEJO DEL ENVÍO DEL FORMULARIO ---
    form.addEventListener('submit', async (e) => {
        e.preventDefault(); // Evitar que la página se recargue
        
        // Validación de rangos
        const utsOriginal = parseFloat(document.getElementById('utsOriginal').value);
        const strainOriginal = parseFloat(document.getElementById('strainOriginal').value);
        
        if (utsOriginal < 0.14 || utsOriginal > 0.63) {
            alert('UTS Original debe estar entre 0.14 y 0.63 MPa');
            return;
        }
        
        if (strainOriginal < 116.37 || strainOriginal > 1799.37) {
            alert('Strain Original debe estar entre 116.37 y 1799.37 %');
            return;
        }
        
        setLoading(true);

        const inputData = {
            sampleId: document.getElementById('sampleId').value,
            hsp: parseFloat(hspSlider.value),
            healingTime: parseFloat(healingTimeSlider.value),
            utsOriginal: utsOriginal,
            strainOriginal: strainOriginal,
            peakLogM: parseFloat(document.getElementById('peakLogM').value),
            molecularWeight: parseFloat(document.getElementById('molecularWeight').value),
            contactAngleMean: parseFloat(document.getElementById('contactAngleMean').value),
            contactAngleStd: parseFloat(document.getElementById('contactAngleStd').value),
            ftir: parseFloat(document.getElementById('ftir').value),
            dsc: parseFloat(document.getElementById('dsc').value)
        };

        try {
            // Enviar datos al servidor Flask (ruta /predict)
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(inputData),
            });

            if (!response.ok) throw new Error(`Error del servidor: ${response.status}`);

            const predictions = await response.json();
            if (predictions.error) throw new Error(predictions.error);

            updateUI(predictions);

        } catch (error) {
            console.error('Error al predecir:', error);
            alert(`Error al contactar el modelo: ${error.message}`);
        } finally {
            setLoading(false);
        }
    });

    // --- 4. FUNCIÓN DE ESTADO DE CARGA ---
    function setLoading(isLoading) {
        if (isLoading) {
            buttonText.classList.add('d-none');
            buttonLoader.style.display = 'inline-block';
            predictButton.disabled = true;
        } else {
            buttonText.classList.remove('d-none');
            buttonLoader.style.display = 'none';
            predictButton.disabled = false;
        }
    }

    // --- 5. FUNCIÓN DE ACTUALIZACIÓN DE UI (RESULTADOS) ---
    function updateUI(predictions) {
        // Mostrar contenedor de resultados
        resultsContainer.classList.remove('visually-hidden'); 
        resultsPlaceholder.classList.add('d-none');
        
        // 1. Actualizar eficiencias predichas
        const utsEfficiency = predictions.HE_UTS_Mean;
        const elongationEfficiency = predictions.HE_Elongation_Mean;
        
        resultUtsEfficiency.textContent = `${utsEfficiency.toFixed(1)}%`;
        resultElongationEfficiency.textContent = `${elongationEfficiency.toFixed(1)}%`;
        
        // Añadir clases de color según el valor
        updateEfficiencyColor(resultUtsEfficiency, utsEfficiency);
        updateEfficiencyColor(resultElongationEfficiency, elongationEfficiency);
        
        // 2. Actualizar código QR
        if (predictions.qr_code) {
            qrCodeImg.src = `data:image/png;base64,${predictions.qr_code}`;
            qrCodeImg.classList.remove('d-none');
            qrPlaceholder.classList.add('d-none');
            downloadQrBtn.classList.remove('d-none');
            
            // Guardar el QR globalmente para descarga
            window.currentQRCode = predictions.qr_code;
        } else {
            qrCodeImg.classList.add('d-none');
            qrPlaceholder.classList.remove('d-none');
            downloadQrBtn.classList.add('d-none');
        }
        
        // 3. Scroll suave a resultados
        resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
    
    // --- 6. FUNCIÓN AUXILIAR PARA COLORES DE EFICIENCIA ---
    function updateEfficiencyColor(element, value) {
        // Remover clases existentes
        element.classList.remove('text-success', 'text-warning', 'text-danger', 'text-primary');
        
        // Añadir clase según el valor
        if (value >= 90) {
            element.classList.add('text-success');
        } else if (value >= 70) {
            element.classList.add('text-warning');
        } else if (value >= 50) {
            element.classList.add('text-primary');
        } else {
            element.classList.add('text-danger');
        }
    }
});

// --- 7. FUNCIÓN GLOBAL PARA DESCARGAR QR ---
function downloadQR() {
    if (!window.currentQRCode) {
        alert('No hay código QR para descargar');
        return;
    }
    
    // Crear un enlace temporal para la descarga
    const link = document.createElement('a');
    link.href = `data:image/png;base64,${window.currentQRCode}`;
    link.download = `PLIX_Etiqueta_${new Date().getTime()}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}