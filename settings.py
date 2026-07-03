"""
Application settings and prompt configurations.
Contains the structured text data utilized by the VisionManager to guide AI inferences.
"""

LANGUAGES = [
    "Afrikaans", "Arabic", "Bulgarian", "Catalan", "Chinese (Mandarin)", "Chinese (Taiwan)",
    "Croatian", "Czech", "Danish", "Dutch", "English", "English (UK)", "English (Australia)",
    "English (India)", "Filipino", "Finnish", "French", "German", "Greek", "Hebrew",
    "Hindi", "Hungarian", "Indonesian", "Italian", "Japanese", "Korean", "Latvian",
    "Lithuanian", "Malay", "Norwegian", "Persian", "Polish", "Portuguese (Brazil)",
    "Portuguese (Portugal)", "Romanian", "Russian", "Serbian", "Slovak", "Slovenian",
    "Spanish", "Spanish (Latin America)", "Swedish", "Thai", "Turkish", "Ukrainian", "Vietnamese"
]

# Fully populated 12 modes mapping to Ctrl+F1 through Ctrl+F12
MODES = [
    "Neutral Audio Description",   # Ctrl+F1
    "Family Friendly",             # Ctrl+F2
    "Childish & Playful",          # Ctrl+F3
    "Technical & Objective",       # Ctrl+F4
    "Explicit",                    # Ctrl+F5
    "Explicit and Naughty",        # Ctrl+F6
    "Humorous & Comedic",          # Ctrl+F7
    "Mysterious & Suspenseful",    # Ctrl+F8
    "Poetic & Lyrical",            # Ctrl+F9
    "Cinematic & Epic",            # Ctrl+F10
    "Romantic & Emotional",        # Ctrl+F11
    "Action & Fast-Paced"          # Ctrl+F12
]

PROMPTS = [
    {
        "title": "Brief Desc",
        "prompt": "Describe this scene briefly for narration.", 
        "frames_count": 1, 
        "frames_interval": 0.5
    },
    {
        "title": "Detail Desc",
        "prompt": "Describe this scene in detail.", 
        "frames_count": 1, 
        "frames_interval": 0.5
    },
    {
        "title": "Identify Objects",
        "prompt": "List the main objects and people visible in this scene.", 
        "frames_count": 1, 
        "frames_interval": 0.5
    },
    {
        "title": "Action Sequence",
        "prompt": "Describe the sequence of actions occurring.", 
        "frames_count": 3, 
        "frames_interval": 0.5
    },
    {
        "title": "Scene Setting",
        "prompt": "Describe the overall setting, environment, and atmosphere.", 
        "frames_count": 1, 
        "frames_interval": 0.5
    },
    {
        "title": "Read Text",
        "prompt": "Read and carefully transcribe any visible text, signs, or subtitles in the scene.", 
        "frames_count": 1, 
        "frames_interval": 0.5
    },
    {
        "title": "Cinematic Establishing Shot",
        "prompt": "Set the scene cinematically. Describe the environment, lighting, and atmosphere as a living world. Focus on the spatial layout.", 
        "frames_count": 1, 
        "frames_interval": 0.5
    },
    {
        "title": "Character Focus",
        "prompt": "Focus entirely on the people or creatures. Describe their expressions, body language, and immediate actions fluidly in the present tense.", 
        "frames_count": 3, 
        "frames_interval": 0.5
    },
    {
        "title": "Fluid Action Narration",
        "prompt": "Narrate the unfolding action sequence as a seamless, continuous real-time story. Infer the natural bridging movements between these moments.", 
        "frames_count": 5, 
        "frames_interval": 0.5
    },
    {
        "title": "Cinematic Micro-Story",
        "prompt": "Translate these sequential visual moments into a vivid, continuous, real-time story from beginning to end. Focus on exact physical actions and spatial movements.", 
        "frames_count": 10, 
        "frames_interval": 1.0
    }
]