// Toast functionality
function showToast(message, type = 'primary') {
    // Crear toast dinámicamente si no existe
    let toastContainer = document.querySelector('.toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container position-fixed top-0 end-0 p-3';
        toastContainer.style.zIndex = '1100';
        document.body.appendChild(toastContainer);
    }

    const toastId = 'toast-' + Date.now();
    const toastHTML = `
        <div id="${toastId}" class="toast align-items-center text-bg-${type} border-0" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="d-flex">
                <div class="toast-body">
                    <i class="fas fa-info-circle me-2"></i>
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
        </div>
    `;
    
    toastContainer.insertAdjacentHTML('beforeend', toastHTML);
    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement);
    
    toast.show();
    
    // Remover el toast del DOM después de que se oculte
    toastElement.addEventListener('hidden.bs.toast', () => {
        toastElement.remove();
    });
}

// Smooth scrolling for navigation links
document.addEventListener('DOMContentLoaded', function() {
    // Smooth scrolling para enlaces de ancla
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Navbar background change on scroll
    const navbar = document.querySelector('.navbar');
    window.addEventListener('scroll', function() {
        if (window.scrollY > 100) {
            navbar.classList.add('navbar-scrolled');
        } else {
            navbar.classList.remove('navbar-scrolled');
        }
    });

    // Animation on scroll
    const animateOnScroll = function() {
        const elements = document.querySelectorAll('.card, .tech-card, .feature-card');
        
        elements.forEach(element => {
            const elementTop = element.getBoundingClientRect().top;
            const elementVisible = 150;
            
            if (elementTop < window.innerHeight - elementVisible) {
                element.classList.add('animate__animated', 'animate__fadeInUp');
            }
        });
    };

    // Inicializar animaciones
    window.addEventListener('scroll', animateOnScroll);
    animateOnScroll(); // Ejecutar una vez al cargar

    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    const tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Contador animado para stats
    const animateCounter = (element, target, duration = 2000) => {
        let start = 0;
        const increment = target / (duration / 16);
        const timer = setInterval(() => {
            start += increment;
            if (start >= target) {
                element.textContent = target + (element.textContent.includes('+') ? '+' : '');
                clearInterval(timer);
            } else {
                element.textContent = Math.floor(start) + (element.textContent.includes('+') ? '+' : '');
            }
        }, 16);
    };

    // Observar cuando las stats entran en vista
    const statsObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const statElements = entry.target.querySelectorAll('.display-4');
                statElements.forEach(stat => {
                    const target = parseInt(stat.textContent);
                    if (!isNaN(target)) {
                        animateCounter(stat, target);
                    }
                });
                statsObserver.unobserve(entry.target);
            }
        });
    });

    const statsSection = document.querySelector('.py-5:last-child');
    if (statsSection) {
        statsObserver.observe(statsSection);
    }
});

// Form validation helper
function validateForm(formId) {
    const form = document.getElementById(formId);
    if (!form) return false;
    
    const inputs = form.querySelectorAll('input[required], select[required], textarea[required]');
    let isValid = true;
    
    inputs.forEach(input => {
        input.classList.remove('is-invalid');
        const feedback = input.parentNode.querySelector('.invalid-feedback');
        if (feedback) feedback.remove();
        
        if (!input.value.trim()) {
            input.classList.add('is-invalid');
            const errorDiv = document.createElement('div');
            errorDiv.className = 'invalid-feedback';
            errorDiv.textContent = 'Este campo es requerido.';
            input.parentNode.appendChild(errorDiv);
            isValid = false;
        }
    });
    
    return isValid;
}

// Loading state for buttons
function setLoadingState(buttonId, loading = true) {
    const button = document.getElementById(buttonId);
    if (!button) return;
    
    if (loading) {
        button.disabled = true;
        const originalText = button.innerHTML;
        button.dataset.originalText = originalText;
        button.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Procesando...';
    } else {
        button.disabled = false;
        button.innerHTML = button.dataset.originalText || button.innerHTML;
    }
}

// Theme switcher (opcional)
function toggleTheme() {
    const html = document.documentElement;
    const currentTheme = html.getAttribute('data-bs-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-bs-theme', newTheme);
    
    // Guardar preferencia
    localStorage.setItem('theme', newTheme);
    showToast(`Tema cambiado a ${newTheme === 'dark' ? 'oscuro' : 'claro'}`, 'info');
}

// Cargar tema guardado
document.addEventListener('DOMContentLoaded', function() {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        document.documentElement.setAttribute('data-bs-theme', savedTheme);
    }
});