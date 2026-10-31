"""Labels for CLIP zero-shot query suggestion: common secondhand-marketplace items."""

LABELS = [
    # furniture
    "sofa", "couch", "loveseat", "armchair", "recliner", "office chair", "dining chair", "bar stool",
    "dining table", "coffee table", "side table", "desk", "standing desk", "bookshelf", "dresser",
    "nightstand", "wardrobe", "TV stand", "bed frame", "mattress", "bunk bed", "crib", "bench", "ottoman",
    "shoe rack", "filing cabinet", "floor lamp", "table lamp", "mirror", "rug", "bean bag chair",
    # appliances & home
    "refrigerator", "washing machine", "dryer", "microwave", "dishwasher", "air fryer", "coffee maker",
    "blender", "stand mixer", "toaster", "vacuum cleaner", "air conditioner", "space heater", "fan",
    "humidifier", "air purifier", "ceiling fan", "pressure cooker", "cast iron skillet",
    # electronics
    "laptop", "desktop computer", "computer monitor", "keyboard", "computer mouse", "tablet", "smartphone",
    "television", "soundbar", "speaker", "headphones", "camera", "camera lens", "drone", "video game console",
    "game controller", "printer", "router", "smartwatch", "record player", "guitar amplifier",
    # music & hobby
    "electric guitar", "acoustic guitar", "bass guitar", "piano", "keyboard synthesizer", "drum set",
    "violin", "ukulele", "telescope", "sewing machine", "board game", "lego set", "jigsaw puzzle",
    # sports & outdoors
    "bicycle", "mountain bike", "electric scooter", "skateboard", "snowboard", "skis", "surfboard",
    "kayak", "tent", "backpack", "treadmill", "exercise bike", "dumbbells", "weight bench", "golf clubs",
    "fishing rod", "paddle board", "helmet", "camping stove", "cooler",
    # tools & garden
    "power drill", "circular saw", "tool box", "workbench", "ladder", "lawn mower", "leaf blower",
    "chainsaw", "pressure washer", "wheelbarrow", "patio furniture", "grill", "fire pit", "garden hose",
    "generator", "air compressor", "welder",
    # vehicles & parts
    "car tire", "car wheel rim", "motorcycle", "car seat", "stroller", "roof rack", "trailer hitch",
    # clothing & misc
    "sneakers", "boots", "jacket", "handbag", "wristwatch", "sunglasses", "wedding dress", "suitcase",
    "baby high chair", "playpen", "toy", "doll", "stuffed animal", "painting", "framed art", "vase",
    "clock", "candle holder", "book", "vinyl record", "suitcase", "wheelchair", "walker",
]

# Deduplicate while keeping order.
LABELS = list(dict.fromkeys(LABELS))
