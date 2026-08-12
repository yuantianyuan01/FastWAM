import json
import logging
import os
import random
import re
from collections import defaultdict

log = logging.getLogger(__name__)


def generate_llm_task_from_summaries(summaries: list[str]) -> tuple[str, list[tuple[str, ...]]]:
    from google import genai

    objects_data = []
    short_to_full_id = {}
    category_counters = defaultdict(int)

    for summary_str in summaries:
        match = re.match(
            r"(\S+)\s+\(category=([^)]+)\s+synset=([^)]+)\)\s+center=\(([^)]+)\)\s+on\s+([^,]+),\s+in\s+(.+)",
            summary_str,
        )
        if not match:
            continue

        full_id, category, synset, center_str, on_surface, in_room = match.groups()
        center_coords = [float(x) for x in center_str.split(",")]

        full_id_parts = full_id.split("_")
        room_id = full_id_parts[-1] if len(full_id_parts) >= 5 else None

        base_category = full_id.split("_")[0]
        short_id = f"{base_category}_{category_counters[base_category]}"
        category_counters[base_category] += 1

        short_to_full_id[short_id] = full_id

        objects_data.append(
            {
                "id": short_id,
                "category": category,
                "synset": synset,
                "center": center_coords,
                "room": room_id,
            }
        )

    short_id_to_room = {}
    for obj in objects_data:
        short_id_to_room[obj["id"]] = obj["room"]

    random.shuffle(objects_data)

    system_prompt = """You are a task planner for an embodied household robot.
You are given a JSON list of objects in a scene. Each object includes:
- id (unique identifier; must be used exactly)
- category and synset
- center (x, y, z)
- room (room identifier - objects must be in the SAME room for valid actions)

Your job is to propose ONE high-level, natural household task,
then enumerate ALL possible actions that satisfy that task.

SUPPORTED ACTION TYPES:
1. "pick_place": Pick up one object and place it on/in another object
2. "open": Open a single object (for openable objects like drawers, cabinets, fridges, microwaves, etc.)

CRITICAL CONSTRAINTS:
- For pick_place actions: Both pick and place objects MUST be in the SAME room (have the same room ID).
- For open actions: Only specify openable objects (drawers, cabinets, refrigerators, microwaves, etc.)
- Pick only clearly movable objects that make sense for the task.
- Place only onto or into sensible target objects present in the scene.
- Do NOT invent or modify object IDs.
- You MUST return ALL valid actions that satisfy the task prompt.
- If multiple objects of the same type exist, include ALL combinations.
- Do NOT skip any valid actions - be exhaustive.
- ONLY include semantically reasonable actions for the task (e.g., for "Make a salad", don't include pencils, soap, or wine bottles).

Task Prompt Requirements:
- Describe a SPECIFIC yet natural household goal that clearly implies what objects should be involved.
- Sound conversational, like something a human would say.
- Do NOT use verbs like "put", "place", "move", "pick", "transfer", "open", etc.
- Instead describe the END STATE or goal using nouns and adjectives.
- Be specific enough that the objects involved are clear, but don't name specific object IDs.
- Avoid overly vague goals that could apply to many unrelated objects.
- The prompt should naturally constrain which objects make sense for the task.
- Examples of good task prompts:
  * "Make a salad" → might include opening fridge/drawer + picking vegetables and placing them in a bowl
  * "Prepare breakfast ingredients" → might include opening fridge/cabinet + picking eggs, milk, bread and placing on counter
  * "Organize the desk supplies" → picking pens, pencils, paper and placing in drawers/organizers + opening desk drawers
  * "Store the fresh produce" → opening fridge + picking fruits/vegetables and placing inside
  * "Set up morning coffee" → opening cabinet + picking mug, coffee, placing on counter near coffee maker
  * "Clear the coffee table clutter" → picking magazines, remotes, cups and placing in appropriate locations
  * "Access kitchen ingredients" → opening multiple cabinets and drawers where ingredients are stored
  * "Unload groceries" → opening fridge/pantry + picking grocery items and placing inside
- Examples of bad task prompts:
  * "Prepare a light meal" (too vague - what specific items?)
  * "Put vegetables in a bowl" (uses manipulation verbs)
  * "Open the fridge and place food" (uses manipulation verbs)
  * "Tidy up" (too general - what objects?)
- Think about semantic coherence: a breakfast task should only involve breakfast foods and relevant containers, not random objects like soap or books.
- Tasks that naturally involve opening: breakfast prep, storing food, accessing ingredients, organizing office/desk items, etc.

Output ONLY valid JSON matching the schema below (do NOT include a "relation" field):

{
  "task_prompt": string,
  "actions": [
    {
      "action_type": "pick_place" | "open",
      "pick_object_id": string (only for pick_place),
      "place_target_id": string (only for pick_place),
      "object_id": string (only for open)
    },
    ...
  ]
}

REMEMBER: Include ALL possible actions that satisfy the task prompt within the same room."""

    scene_data = {"scene_objects": objects_data, "total_count": len(objects_data)}

    GENAI_API_KEY = os.environ.get("GENAI_API_KEY")
    if not GENAI_API_KEY:
        raise ValueError(
            "GENAI_API_KEY environment variable not set. Please set it to your Google GenAI API key."
        )
    client = genai.Client(api_key=GENAI_API_KEY)

    user_message = f"Scene data:\n{json.dumps(scene_data, indent=2)}"
    full_prompt = f"{system_prompt}\n\n{user_message}"

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite", contents=full_prompt, config={"temperature": 0.5}
    )

    response_text = response.text.strip()
    if response_text.startswith("```"):
        response_text = response_text.split("```")[1]
        if response_text.startswith("json"):
            response_text = response_text[4:]
        response_text = response_text.strip()

    result = json.loads(response_text)

    actions_with_full_ids = []
    for action in result["actions"]:
        action_type = action.get("action_type", "pick_place")

        if action_type == "pick_place":
            short_pick_id = action.get("pick_object_id")
            short_place_id = action.get("place_target_id")

            if not short_pick_id or not short_place_id:
                log.warning("Missing pick or place object ID in action, skipping")
                continue

            if short_pick_id not in short_to_full_id:
                log.warning(f"Pick object ID '{short_pick_id}' not found in scene, skipping action")
                continue
            if short_place_id not in short_to_full_id:
                log.warning(
                    f"Place object ID '{short_place_id}' not found in scene, skipping action"
                )
                continue

            pick_room = short_id_to_room.get(short_pick_id)
            place_room = short_id_to_room.get(short_place_id)

            if pick_room != place_room:
                log.warning(
                    f"Objects not in same room: {short_pick_id} (room {pick_room}) and "
                    f"{short_place_id} (room {place_room}), skipping action"
                )
                continue

            full_pick_id = short_to_full_id[short_pick_id]
            full_place_id = short_to_full_id[short_place_id]

            actions_with_full_ids.append(("pick_place", full_pick_id, full_place_id))

        elif action_type == "open":
            short_object_id = action.get("object_id")

            if not short_object_id:
                log.warning("Missing object ID in open action, skipping")
                continue

            if short_object_id not in short_to_full_id:
                log.warning(f"Object ID '{short_object_id}' not found in scene, skipping action")
                continue

            full_object_id = short_to_full_id[short_object_id]

            actions_with_full_ids.append(("open", full_object_id))
        else:
            log.warning(f"Unknown action type '{action_type}', skipping action")
            continue

    if len(actions_with_full_ids) == 0:
        log.warning(f"No valid actions after filtering for task: {result['task_prompt']}")

    return (result["task_prompt"], actions_with_full_ids)
