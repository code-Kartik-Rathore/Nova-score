"""
Animation and micro-interaction utilities for the Nova Score application.
"""
import streamlit as st
from streamlit.components.v1 import html
import time

def animate_score_counter(container, target_score, duration=1.5):
    """
    Animate a counter from 0 to the target score.
    
    Args:
        container: Streamlit container to display the counter in
        target_score: The final score to count up to
        duration: Duration of the animation in seconds
    """
    import time
    
    # Create a placeholder for the counter
    counter_placeholder = container.empty()
    
    # Calculate the number of steps
    steps = min(60, int(duration * 30))  # Max 60 steps for smoothness
    step_size = target_score / steps
    
    # Animate the counter
    for i in range(steps + 1):
        current_score = int(step_size * i)
        counter_placeholder.markdown(
            f"<h1 style='text-align: center;'>{min(current_score, target_score)}</h1>",
            unsafe_allow_html=True
        )
        time.sleep(duration / steps)
    
    return counter_placeholder

def add_loading_animation():
    """Add a loading animation to the page."""
    return st.spinner("Processing...")

def add_success_message(message, delay=2):
    """Display a temporary success message."""
    success = st.success(message)
    time.sleep(delay)
    success.empty()

def add_error_message(message, delay=3):
    """Display a temporary error message."""
    error = st.error(message)
    time.sleep(delay)
    error.empty()

def add_warning_message(message, delay=3):
    """Display a temporary warning message."""
    warning = st.warning(message)
    time.sleep(delay)
    warning.empty()

def add_info_message(message, delay=3):
    """Display a temporary info message."""
    info = st.info(message)
    time.sleep(delay)
    info.empty()

def pulse_element(key, color="#4CAF50"):
    """Add a pulsing animation to an element."""
    return f"""
    <style>
        @keyframes pulse-{key} {{
            0% {{ transform: scale(1); }}
            50% {{ transform: scale(1.05); }}
            100% {{ transform: scale(1); }}
        }}
        .pulse-{key} {{
            animation: pulse-{key} 1.5s infinite;
            color: {color};
        }}
    </style>
    <div class="pulse-{key}">
        {st.session_state.get(f'pulse_{key}_content', '')}
    </div>
    """

def add_confetti():
    """Add a confetti animation to celebrate success."""
    confetti_html = """
    <canvas id="confetti-canvas"></canvas>
    <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.5.1/dist/confetti.browser.min.js"></script>
    <script>
        const duration = 3000;
        const animationEnd = Date.now() + duration;
        const defaults = { startVelocity: 30, spread: 360, ticks: 60, zIndex: 0 };
        
        function randomInRange(min, max) {
            return Math.random() * (max - min) + min;
        }
        
        const interval = setInterval(function() {
            const timeLeft = animationEnd - Date.now();
            
            if (timeLeft <= 0) {
                return clearInterval(interval);
            }
            
            const particleCount = 50 * (timeLeft / duration);
            confetti({
                ...defaults,
                particleCount,
                origin: { x: randomInRange(0.1, 0.3), y: Math.random() - 0.2 }
            });
            confetti({
                ...defaults,
                particleCount,
                origin: { x: randomInRange(0.7, 0.9), y: Math.random() - 0.2 }
            });
        }, 250);
    </script>
    """
    html(confetti_html, height=0)
