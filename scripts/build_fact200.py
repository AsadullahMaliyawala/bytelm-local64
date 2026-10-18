"""Build data/eval/fact200.jsonl and hash-lock it.

CPU only. Drops the file unless every question is absent from the SFT text
and from the templated Step 7 questions. Few-shot examples used at score
time are not part of the set.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "eval" / "fact200.jsonl"
LOCK = ROOT / "data" / "eval" / "fact200.lock.json"

FEW_SHOT = {
    "what is the capital of japan?",
    "what color is the sky on a clear day?",
    "what is 3 + 4?",
    "who painted the mona lisa?",
    "what do bees make?",
}

# (category, question, gold, aliases including gold)
ITEMS: list[tuple[str, str, str, list[str]]] = []


def add(category: str, question: str, gold: str, aliases: list[str]) -> None:
    extra = [gold] + list(aliases)
    seen: list[str] = []
    for alias in extra:
        if alias not in seen:
            seen.append(alias)
    ITEMS.append((category, question, gold, seen))


for country, gold, aliases in [
    ("Chile", "Santiago", []),
    ("the Philippines", "Manila", []),
    ("Iran", "Tehran", []),
    ("Iraq", "Baghdad", []),
    ("Cambodia", "Phnom Penh", []),
    ("Laos", "Vientiane", []),
    ("Bhutan", "Thimphu", ["Thimpu"]),
    ("Uzbekistan", "Tashkent", []),
    ("Ecuador", "Quito", []),
    ("Venezuela", "Caracas", []),
    ("Uruguay", "Montevideo", []),
    ("Paraguay", "Asuncion", ["Asunción"]),
    ("Costa Rica", "San Jose", ["San José"]),
    ("Panama", "Panama City", []),
    ("Guatemala", "Guatemala City", []),
    ("Honduras", "Tegucigalpa", []),
    ("Nicaragua", "Managua", []),
    ("the Dominican Republic", "Santo Domingo", []),
    ("Jamaica", "Kingston", []),
    ("Senegal", "Dakar", []),
    ("Tanzania", "Dodoma", []),
    ("Uganda", "Kampala", []),
    ("Zimbabwe", "Harare", []),
    ("Zambia", "Lusaka", []),
    ("Mozambique", "Maputo", []),
    ("Angola", "Luanda", []),
    ("Tunisia", "Tunis", []),
    ("Libya", "Tripoli", []),
    ("Sudan", "Khartoum", []),
    ("Oman", "Muscat", []),
    ("Armenia", "Yerevan", []),
    ("Azerbaijan", "Baku", []),
    ("Croatia", "Zagreb", []),
    ("Serbia", "Belgrade", []),
    ("Bulgaria", "Sofia", []),
    ("Slovakia", "Bratislava", []),
    ("Estonia", "Tallinn", []),
    ("Latvia", "Riga", []),
    ("Lithuania", "Vilnius", []),
    ("Slovenia", "Ljubljana", []),
]:
    add("capitals", f"What is the capital of {country}?", gold, aliases)

for question, gold, aliases in [
    ("How many planets are in the solar system?", "8", ["eight"]),
    ("Which planet is known for its rings?", "Saturn", []),
    ("What is the closest star to Earth?", "the Sun", ["Sun", "Sol"]),
    ("What gas do humans need to breathe in?", "oxygen", ["O2"]),
    ("What gas do humans breathe out from their lungs?", "carbon dioxide", ["CO2"]),
    ("What is the chemical symbol for gold?", "Au", []),
    ("What is the chemical formula for water?", "H2O", []),
    ("At what temperature Celsius does water freeze?", "0", ["0°C", "0 C", "zero"]),
    ("How many bones does an adult human skeleton have?", "206", []),
    ("Which organ pumps blood through the body?", "the heart", ["heart"]),
    ("What is the hardest natural mineral?", "diamond", []),
    ("Which atomic particle carries a negative charge?", "electron", ["electrons"]),
    ("How many chromosomes are in a human body cell?", "46", []),
    ("What is the largest organ of the human body?", "skin", ["the skin"]),
    ("Which metal is liquid at room temperature?", "mercury", []),
    ("What is the boiling point of water in degrees Celsius?", "100", ["100°C", "100 C"]),
    ("What gas makes up most of Earth's atmosphere?", "nitrogen", []),
    ("How many moons does Earth have?", "1", ["one"]),
    ("What is the center of an atom called?", "nucleus", ["the nucleus"]),
    ("Which planet is closest to the Sun?", "Mercury", []),
    ("How many chambers does the human heart have?", "4", ["four"]),
    ("Which vitamin does skin make from sunlight?", "vitamin D", []),
    ("What is often called the powerhouse of the cell?", "mitochondria", ["mitochondrion"]),
    ("How many elements are on the periodic table?", "118", []),
    ("At what temperature Fahrenheit does water freeze?", "32", ["32°F", "32 F"]),
    ("Which planet is called the Red Planet?", "Mars", []),
    ("What is the chemical symbol for sodium?", "Na", []),
    ("What is the chemical symbol for iron?", "Fe", []),
    ("What gas do plants release during photosynthesis?", "oxygen", ["O2"]),
    ("What is the chemical symbol for silver?", "Ag", []),
    ("What is the unit of electrical resistance?", "ohm", ["Ohm"]),
    ("How many legs does an insect have?", "6", ["six"]),
    ("What galaxy contains Earth?", "the Milky Way", ["Milky Way"]),
    ("Which planet is the smallest in the solar system?", "Mercury", []),
    ("What color is chlorophyll?", "green", []),
    ("What instrument measures temperature?", "thermometer", ["a thermometer"]),
    ("How many teeth does a typical adult human have?", "32", []),
    ("What is the largest planet in the solar system?", "Jupiter", []),
    ("About how many days does Earth take to orbit the Sun?", "365", ["365 days", "one year", "1 year", "a year"]),
    ("What is the chemical symbol for helium?", "He", []),
]:
    add("science", question, gold, aliases)

for question, gold, aliases in [
    ("Who wrote Hamlet?", "William Shakespeare", ["Shakespeare"]),
    ("Who wrote Don Quixote?", "Miguel de Cervantes", ["Cervantes"]),
    ("Who painted The Starry Night?", "Vincent van Gogh", ["van Gogh", "Van Gogh"]),
    ("Who wrote Pride and Prejudice?", "Jane Austen", ["Austen"]),
    ("Who composed the opera The Magic Flute?", "Wolfgang Amadeus Mozart", ["Mozart"]),
    ("Who composed the Moonlight Sonata?", "Ludwig van Beethoven", ["Beethoven"]),
    ("Who discovered penicillin?", "Alexander Fleming", ["Fleming"]),
    ("Who wrote The Odyssey?", "Homer", []),
    ("Who painted The Persistence of Memory?", "Salvador Dali", ["Salvador Dalí", "Dali", "Dalí"]),
    ("Who wrote the novel 1984?", "George Orwell", ["Orwell"]),
    ("Who developed the theory of relativity?", "Albert Einstein", ["Einstein"]),
    ("Who wrote Frankenstein?", "Mary Shelley", ["Shelley"]),
    ("Who composed The Four Seasons?", "Antonio Vivaldi", ["Vivaldi"]),
    ("Who wrote Moby-Dick?", "Herman Melville", ["Melville"]),
    ("Who painted The Birth of Venus?", "Sandro Botticelli", ["Botticelli"]),
    ("Who wrote Crime and Punishment?", "Fyodor Dostoevsky", ["Dostoevsky", "Dostoyevsky"]),
    ("Who painted the ceiling of the Sistine Chapel?", "Michelangelo", []),
    ("Who wrote One Hundred Years of Solitude?", "Gabriel Garcia Marquez", ["García Márquez", "Garcia Marquez"]),
    ("Who wrote The Divine Comedy?", "Dante Alighieri", ["Dante"]),
    ("Who was British prime minister for most of World War II?", "Winston Churchill", ["Churchill"]),
    ("Who wrote To Kill a Mockingbird?", "Harper Lee", []),
    ("Who wrote The Origin of Species?", "Charles Darwin", ["Darwin"]),
    ("Who wrote Les Miserables?", "Victor Hugo", ["Hugo"]),
    ("Who composed Swan Lake?", "Pyotr Tchaikovsky", ["Tchaikovsky", "Tschaikovsky"]),
    ("Who was the 16th president of the United States?", "Abraham Lincoln", ["Lincoln"]),
    ("Who painted Guernica?", "Pablo Picasso", ["Picasso"]),
    ("Who wrote The Iliad?", "Homer", []),
    ("Who painted Girl with a Pearl Earring?", "Johannes Vermeer", ["Vermeer"]),
    ("Who invented the practical telephone?", "Alexander Graham Bell", ["Graham Bell", "Bell"]),
    ("Who wrote A Christmas Carol?", "Charles Dickens", ["Dickens"]),
]:
    add("people", question, gold, aliases)

for question, gold in [
    ("What is 8 + 5?", "13"),
    ("What is 9 * 9?", "81"),
    ("What is 20 - 7?", "13"),
    ("What is 100 / 4?", "25"),
    ("What is 15 + 6?", "21"),
    ("What is 7 * 8?", "56"),
    ("What is 50 - 18?", "32"),
    ("What is 81 / 9?", "9"),
    ("What is 14 + 9?", "23"),
    ("What is 6 * 7?", "42"),
    ("What is 30 - 12?", "18"),
    ("What is 64 / 8?", "8"),
    ("What is 11 + 11?", "22"),
    ("What is 4 * 12?", "48"),
    ("What is 90 - 25?", "65"),
    ("What is 36 / 6?", "6"),
    ("What is 13 + 8?", "21"),
    ("What is 5 * 5?", "25"),
    ("What is 40 - 9?", "31"),
    ("What is 72 / 8?", "9"),
    ("What is 16 + 7?", "23"),
    ("What is 3 * 9?", "27"),
    ("What is 60 - 15?", "45"),
    ("What is 48 / 6?", "8"),
    ("What is 19 + 4?", "23"),
    ("What is 8 * 8?", "64"),
    ("What is 25 - 9?", "16"),
    ("What is 45 / 5?", "9"),
    ("What is 17 + 6?", "23"),
    ("What is 9 * 6?", "54"),
]:
    add("arithmetic", question, gold, [gold])

for question, gold, aliases in [
    ("What is the largest ocean on Earth?", "the Pacific Ocean", ["Pacific Ocean", "Pacific"]),
    ("What is the highest mountain on Earth?", "Mount Everest", ["Everest"]),
    ("What is the smallest continent?", "Australia", []),
    ("What ocean lies between Europe and the Americas?", "the Atlantic Ocean", ["Atlantic Ocean", "Atlantic"]),
    ("What is the largest country by area?", "Russia", []),
    ("What is the longest river in Africa?", "the Nile", ["Nile"]),
    ("What sea lies between Europe and Africa?", "the Mediterranean Sea", ["Mediterranean Sea", "Mediterranean"]),
    ("In which country is the Eiffel Tower?", "France", []),
    ("What river runs through London?", "the Thames", ["Thames", "River Thames"]),
    ("What is the largest lake in Africa?", "Lake Victoria", ["Victoria"]),
    ("What mountain range contains Mount Everest?", "the Himalayas", ["Himalayas"]),
    ("What is the largest island in the world?", "Greenland", []),
    ("What continent is Brazil in?", "South America", []),
    ("What is the smallest country in the world by area?", "Vatican City", ["the Vatican", "Vatican"]),
    ("What ocean is east of Africa?", "the Indian Ocean", ["Indian Ocean", "Indian"]),
    ("What is the largest hot desert?", "the Sahara", ["Sahara", "Sahara Desert"]),
    ("What line divides Earth into northern and southern hemispheres?", "the equator", ["equator"]),
    ("What country is Mount Fuji in?", "Japan", []),
    ("What is the southernmost continent?", "Antarctica", []),
    ("What is the largest coral reef system?", "the Great Barrier Reef", ["Great Barrier Reef"]),
    ("What country is the Taj Mahal in?", "India", []),
    ("What river flows through Paris?", "the Seine", ["Seine"]),
    ("What is the largest U.S. state by area?", "Alaska", []),
    ("What canal connects the Atlantic and Pacific oceans?", "the Panama Canal", ["Panama Canal"]),
    ("On which continent is the Sahara Desert?", "Africa", []),
    ("What country is the Great Wall in?", "China", []),
    ("What continent is Japan in?", "Asia", []),
    ("What is the longest river in South America?", "the Amazon", ["Amazon", "Amazon River"]),
    ("What waterfall in Venezuela is the world's highest?", "Angel Falls", []),
    ("What is the largest rainforest?", "the Amazon rainforest", ["Amazon", "Amazon rainforest"]),
]:
    add("geography", question, gold, aliases)

for question, gold, aliases in [
    ("How many days are in a leap year?", "366", []),
    ("How many hours are in a day?", "24", ["twenty-four"]),
    ("How many months have 31 days?", "7", ["seven"]),
    ("What color do you get by mixing red and yellow?", "orange", []),
    ("How many sides does a hexagon have?", "6", ["six"]),
    ("What is the opposite of up?", "down", []),
    ("How many letters are in the English alphabet?", "26", ["twenty-six"]),
    ("What day comes after Monday?", "Tuesday", []),
    ("How many cents are in a dollar?", "100", []),
    ("What is the first month of the year?", "January", []),
    ("How many wheels does a tricycle have?", "3", ["three"]),
    ("What is frozen water called?", "ice", []),
    ("How many seconds are in one minute?", "60", ["sixty"]),
    ("What is the opposite of empty?", "full", []),
    ("How many colors are in a rainbow?", "7", ["seven"]),
    ("What meal is usually eaten in the morning?", "breakfast", []),
    ("How many legs does a typical chair have?", "4", ["four"]),
    ("How many players from one team are on a soccer field?", "11", ["eleven"]),
    ("What color is a ripe banana?", "yellow", []),
    ("What is a baby dog called?", "a puppy", ["puppy"]),
    ("How many sides does a stop sign have?", "8", ["eight"]),
    ("What is the opposite of dark?", "light", []),
    ("What fruit is traditionally used to make wine?", "grapes", ["grape"]),
    ("How many seasons are there in a year?", "4", ["four"]),
    ("What is the last month of the year?", "December", []),
    ("How many inches are in a foot?", "12", ["twelve"]),
    ("What tool is used to cut paper?", "scissors", []),
    ("How many days are in January?", "31", []),
    ("How many keys does a standard piano have?", "88", []),
    ("What day comes before Friday?", "Thursday", []),
]:
    add("everyday", question, gold, aliases)


def corpus() -> tuple[str, list[str]]:
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from lct.data.sft_v7 import CAPITALS, DEFINITIONS, YES_NO

    chunks: list[str] = []
    used = [
        "data/sft_v7/train.jsonl",
        "data/sft_v7/holdout.json",
        "data/sft_v7/short_questions.json",
        "data/sft_v7/detail_prompts.json",
        "data/sft/holdout_text.json",
    ]
    for rel in used:
        chunks.append(Path(ROOT / rel).read_text(encoding="utf-8").casefold())
    for country, capital in CAPITALS:
        chunks.append(f"what is the capital of {country}? the capital of {country} is {capital}.".casefold())
    for question, answer in YES_NO:
        chunks.append(f"{question} {answer}".casefold())
    for topic, answer in DEFINITIONS:
        chunks.append(f"what is {topic}? {answer}".casefold())
    teacher = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.casefold()
        if "teacher" not in name:
            continue
        if path.suffix.casefold() not in {".jsonl", ".json", ".txt", ".md"}:
            continue
        if "checkpoint" in path.parts:
            continue
        teacher.append(str(path.relative_to(ROOT)).replace("\\", "/"))
        chunks.append(path.read_text(encoding="utf-8", errors="ignore").casefold())
    return "\n".join(chunks), teacher


def main() -> None:
    counts: dict[str, int] = {}
    for category, _q, _g, _a in ITEMS:
        counts[category] = counts.get(category, 0) + 1
    expect = {"capitals": 40, "science": 40, "people": 30, "arithmetic": 30, "geography": 30, "everyday": 30}
    if counts != expect or len(ITEMS) != 200:
        raise SystemExit(f"bad counts {counts} n={len(ITEMS)}")
    questions = [row[1].casefold() for row in ITEMS]
    if len(set(questions)) != 200:
        raise SystemExit("duplicate questions")
    for question in FEW_SHOT:
        if question in questions:
            raise SystemExit(f"few-shot leaked into the set: {question}")
    blob, teacher = corpus()
    hits = []
    for category, question, _gold, _aliases in ITEMS:
        needle = question.casefold()
        if needle in blob or needle.rstrip("?") in blob:
            hits.append(question)
    if hits:
        raise SystemExit("overlap:\n" + "\n".join(hits))
    prefixes = {"capitals": "cap", "science": "sci", "people": "ppl", "arithmetic": "math", "geography": "geo", "everyday": "day"}
    seen_n = {key: 0 for key in prefixes}
    lines = []
    for category, question, gold, aliases in ITEMS:
        seen_n[category] += 1
        row = {
            "id": f"{prefixes[category]}-{seen_n[category]:02d}",
            "category": category,
            "question": question,
            "gold": gold,
            "aliases": aliases,
        }
        lines.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
    text = "\n".join(lines) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8", newline="\n")
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    lock = {
        "file": "data/eval/fact200.jsonl",
        "sha256": digest,
        "n": 200,
        "categories": expect,
        "excluded_against": [
            "data/sft_v7/train.jsonl",
            "data/sft_v7/holdout.json",
            "data/sft_v7/short_questions.json",
            "data/sft_v7/detail_prompts.json",
            "data/sft/holdout_text.json",
            "templated capitals, yes/no, and definitions in src/lct/data/sft_v7.py",
        ],
        "teacher_pipeline_files": teacher,
    }
    LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8", newline="\n")
    print({"wrote": str(OUT), "sha256": digest, "teacher_pipeline_files": teacher}, flush=True)


if __name__ == "__main__":
    main()
