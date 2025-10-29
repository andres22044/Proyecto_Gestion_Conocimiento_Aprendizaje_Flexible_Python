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
    const gaugeUts = document.getElementById('gauge-uts-fill');
    const gaugeUtsLabel = document.getElementById('gauge-uts-label');
    const resultUtsLabel = document.getElementById('result-uts-label');
    const gaugeElongation = document.getElementById('gauge-elongation-fill');
    const gaugeElongationLabel = document.getElementById('gauge-elongation-label');
    const resultElongationLabel = document.getElementById('result-elongation-label');
    const resultMaxStress = document.getElementById('result-max-stress');
    const resultMaxStrain = document.getElementById('result-max-strain');
    const chartImage = document.getElementById('prediction-chart-img');
    const chartPlaceholder = document.getElementById('chart-placeholder');

    // --- 2. ACTUALIZACIÓN DE VALORES DE SLIDERS ---
    hspSlider.addEventListener('input', (e) => {
        hspValue.textContent = parseFloat(e.target.value).toFixed(2);
    });
    healingTimeSlider.addEventListener('input', (e) => {
        healingTimeValue.textContent = `${parseFloat(e.target.value).toFixed(2)} h`;
    });

    // --- 3. MANEJO DEL ENVÍO DEL FORMULARIO ---
    form.addEventListener('submit', async (e) => {
        e.preventDefault(); // Evitar que la página se recargue
        
        setLoading(true);

        const inputData = {
            hsp: parseFloat(hspSlider.value),
            healingTime: parseFloat(healingTimeSlider.value),
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
            buttonText.classList.add('d-none'); // d-none es display:none en Bootstrap
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
        resultsContainer.classList.remove('visually-hidden'); // Quita el oculto
        resultsPlaceholder.classList.add('d-none'); // Oculta el placeholder
        
        updateGauge(gaugeUts, gaugeUtsLabel, resultUtsLabel, predictions.HE_UTS_Mean);
        updateGauge(gaugeElongation, gaugeElongationLabel, resultElongationLabel, predictions.HE_Elongation_Mean);
        
        resultMaxStress.textContent = `${predictions.Max_Stress.toFixed(2)} MPa`;
        resultMaxStrain.textContent = `${predictions.Max_Strain.toFixed(1)} %`;

        if (predictions.chart_image_url) {
            chartImage.src = predictions.chart_image_url; 
            chartImage.classList.remove('d-none'); // Mostrar la imagen
            chartPlaceholder.classList.add('d-none'); // Ocultar el texto
        } else {
            chartImage.classList.add('d-none');
            chartPlaceholder.classList.remove('d-none');
        }
    }
    
    // --- 6. FUNCIÓN AUXILIAR PARA MEDIDORES ---
    function updateGauge(gaugeElement, textElement, labelElement, percentage) {
        if (percentage < 0) percentage = 0;
        if (percentage > 100) percentage = 100;
        
        const angle = (percentage / 100) * 180;
        gaugeElement.style.transform = `rotate(${angle}deg)`;
        textElement.textContent = `${percentage.toFixed(1)}%`;
        
        // Colores de Bootstrap
        const textSuccess = 'text-success';
        const textWarning = 'text-warning';
        const textDanger = 'text-danger';
        const bgSuccess = 'bg-success';
        const bgWarning = 'bg-warning';
        const bgDanger = 'bg-danger';

        // Quitar clases antiguas
        gaugeElement.classList.remove(bgSuccess, bgWarning, bgDanger);
        labelElement.classList.remove(textSuccess, textWarning, textDanger);

        if (percentage >= 90) {
            gaugeElement.classList.add(bgSuccess);
            labelElement.textContent = 'ÓPTIMO';
            labelElement.classList.add(textSuccess);
        } else if (percentage >= 70) {
            gaugeElement.classList.add(bgWarning);
            labelElement.textContent = 'BUENO';
            labelElement.classList.add(textWarning);
        } else {
            gaugeElement.classList.add(bgDanger);
            labelElement.textContent = 'BAJO';
            labelElement.classList.add(textDanger);
        }
    }
});
