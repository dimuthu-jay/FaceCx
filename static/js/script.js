
particlesJS("particles-js", {
    "particles": {
        "number": { "value": 80 },
        "size": { "value": 4, "random": true },
        "color": { "value": "#a855f7",
            //  "opacity": 0.1
        },
        "line_linked": {
            "enable": true,
            "distance": 150,
            "color": "#8d289a",
            // "opacity": 0.1,
        },
        "move": {
            "speed": 2
        },
        "interactivity": {
            "events": {
                "onhover": { "enable": true, "mode": "grab" },
                "onclick": { "enable": true, "mode": "push" }
            },
            "modes": {
                "grab": { "distance": 200, "line_linked": { "opacity": 1 } },
                "push": { "particles_nb": 4 }
            }
        }
    }
});

