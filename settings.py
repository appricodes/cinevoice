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

# 11 modes mapping to Ctrl+F1 through Ctrl+F11 (Ctrl+F12 edits the custom instruction instead)
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
        "frames_count": 9, 
        "frames_interval": 1.0
    },
    {
        "title": "Character Focus",
        "prompt": "Focus entirely on the people or creatures. Describe their expressions, body language, and immediate actions fluidly in the present tense.", 
        "frames_count": 9, 
        "frames_interval": 1.0
    },
    {
        "title": "Fluid Action Narration",
        "prompt": "Narrate what happens across this moment as a short, chronological story, in the order it unfolds. Add a touch of sensory detail -- setting, mood, expression -- so the listener can picture it, but skip trivial motions like a glance or a small gesture shift.",
        "frames_count": 9,
        "frames_interval": 1.0
    },
    {
        "title": "Cinematic Micro-Story",
        "prompt": "Tell what happens across these moments as a vivid, chronological micro-story, from beginning to end. Include enough sensory and atmospheric detail -- surroundings, mood, who's involved -- to help the listener picture the scene, without dwelling on minor or valueless movements.",
        "frames_count": 18,
        "frames_interval": 6.0
    }
]