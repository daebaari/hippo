"""Benchmark relation-classification backends on a fixed labeled fixture set.

Compares:
- qwen-32b-mlx       (current production LocalLLM)
- gemma-4-26b-moe    (lmstudio-community MLX 4-bit)
- gemini-3-flash     (current production GeminiLLM)

Also includes throughput experiments on Gemma:
- gemma-control      (heads-at-end prompt, no batching, no prefix cache)
- gemma-batch        (mlx_lm.batch_generate, configurable batch size)
- gemma-prefix       (prefilled prefix KV cache reused per call)

Reports per-backend accuracy and latency stats on the simplified 3-class
taxonomy (causes / contradicts / related / none).

Run:
    uv run python scripts/bench_relation_classifiers.py
    uv run python scripts/bench_relation_classifiers.py --backends qwen,gemma
    uv run python scripts/bench_relation_classifiers.py --skip-heavy   # gemini only
    uv run python scripts/bench_relation_classifiers.py --backends gemma-control,gemma-batch,gemma-prefix
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

# === Shared 3-class prompt (the simplified taxonomy from brainstorming) ===
SIMPLIFIED_PROMPT = '''You are deciding whether two memory heads are related, and if so, by what relation.

Head A: "{head_a}"
Head B: "{head_b}"

(Both heads point to bodies in the same memory store. They surfaced as candidates because their embeddings are similar.)

Possible relations:
- "causes" — A causes B (or B is a consequence of A). Asymmetric.
- "contradicts" — A and B make incompatible claims. Symmetric.
- "related" — meaningfully connected but no specific relation. Symmetric.
- "none" — not actually related; the embedding similarity was misleading.

Output a single JSON object: {{"relation": "<one of the above>", "weight": <float 0-1>, "reason": "<one sentence>"}}. Return ONLY the JSON.'''


# === Heads-at-end variant for prefix-cache experiments ===
# All static instructions live before the variable heads, maximizing the cacheable
# prefix length. Used by gemma-control / gemma-batch / gemma-prefix so that the
# three throughput strategies are compared on identical inputs.
PREFIX_OPTIMIZED_PROMPT = '''You are deciding whether two memory heads are related, and if so, by what relation.

(Both heads point to bodies in the same memory store. They surfaced as candidates because their embeddings are similar.)

Possible relations:
- "causes" — A causes B (or B is a consequence of A). Asymmetric.
- "contradicts" — A and B make incompatible claims. Symmetric.
- "related" — meaningfully connected but no specific relation. Symmetric.
- "none" — not actually related; the embedding similarity was misleading.

Output a single JSON object: {{"relation": "<one of the above>", "weight": <float 0-1>, "reason": "<one sentence>"}}. Return ONLY the JSON.

Head A: "{head_a}"
Head B: "{head_b}"'''


# === Fixture set: tricky-but-clear pairs, relabeled under 3-class taxonomy ===
FIXTURES: dict[str, list[tuple[str, str]]] = {
    'contradicts': [
        ('Reranker is resident in the memory daemon and loaded at startup',
         'Reranker is invoked via a subprocess spawned per request'),
        ('Stop hook must complete before the user session is released',
         'Stop hook spawns a detached background subprocess and returns immediately'),
        ('Edge weights are independent per relation, in range zero to one',
         'Edge weights for a head sum to one across all outgoing relations'),
        ('The memory daemon shuts down cleanly when launchd sends SIGTERM',
         'The memory daemon ignores SIGTERM and only exits on SIGKILL'),
        ('Captured turns are filed under the project basename of the cwd ancestor',
         'Captured turns are filed under the user home directory regardless of cwd'),
        ('Embedding is computed in-process via sentence-transformers',
         'Embedding is requested over a Unix socket from the memory daemon'),
        ('Capture queue rows include the full assistant turn text',
         'Capture queue rows include only a summary written at capture time'),
        ('Lock files are zero-byte sentinels removed on process exit',
         'Lock files are JSON with PID and timestamp, swept by SessionStart hook'),
        ('Edge proposal stores only the relation type',
         'Edge proposal stores relation type, weight, and proposer reasoning'),
        ('Heavy dream runs atomize then edge proposal then contradiction resolution',
         'Heavy dream runs atomize then review then multi-head then edge proposal then contradiction'),
        # +5 expansion
        ('Vector search returns hits ordered by ascending distance',
         'Vector search returns hits in arbitrary order and callers must sort the result'),
        ('The memory daemon serializes all requests through a single connection',
         'The memory daemon accepts concurrent client connections via a thread pool'),
        ('All edges in the graph carry an explicit relation type field',
         'Edges are stored as untyped links with no relation metadata recorded'),
        ('Capture queue rows are retained indefinitely until explicit deletion',
         'Capture queue rows older than the retention threshold are auto-purged at startup'),
        ('The atomize prompt outputs structured JSON with explicit named fields',
         'The atomize prompt produces free-form prose summaries with no schema'),
        # === Expansion: 250 diverse pairs ===
        ('Adding caffeine reduces sleep latency in healthy adults by several minutes',
         'Drinking caffeinated coffee within an hour of bedtime lengthens the time required to fall asleep'),
        ('The Great Wall of China is visible to the naked eye from low Earth orbit',
         'Astronauts have repeatedly confirmed that the Great Wall of China cannot be distinguished without aid from orbit'),
        ('Bats navigate primarily by echolocation rather than by visual cues at night',
         'Bats rely chiefly on their excellent eyesight to find prey during nighttime flights'),
        ('Antibiotics are effective treatments for bacterial infections of the throat',
         'Antibiotics will clear up a typical viral cold within a few days of starting them'),
        ('The human body contains around two hundred and six bones at adult maturity',
         'Adult humans typically have just over three hundred bones in their fully developed skeleton'),
        ('Mount Everest stands roughly eight thousand eight hundred meters above sea level',
         'Mount Everest measures approximately six thousand meters from base to summit at sea level'),
        ('Light travels at roughly three hundred thousand kilometers per second in vacuum',
         'Light moves through empty space at about thirty thousand kilometers per second'),
        ('The Pacific Ocean is the largest ocean on Earth by surface area',
         'The Atlantic Ocean covers a greater surface area than any other ocean on the planet'),
        ('Water boils at one hundred degrees Celsius at standard atmospheric pressure',
         'At sea-level atmospheric pressure water reaches its boiling point near eighty degrees Celsius'),
        ('Pluto was reclassified as a dwarf planet by the International Astronomical Union in two thousand six',
         'Pluto retains full planetary status under the current classification used by astronomers worldwide'),
        ('The Amazon rainforest produces a substantial fraction of the oxygen in the atmosphere',
         'The Amazon rainforest is essentially a net consumer of oxygen and contributes negligibly to atmospheric supply'),
        ('Sharks are fish that breathe by drawing oxygen from water through their gills',
         'Sharks are warm-blooded marine mammals that surface periodically to breathe air'),
        ('The Berlin Wall fell in nineteen eighty-nine after decades of dividing the city',
         'The Berlin Wall was demolished in nineteen seventy-two as part of postwar reunification'),
        ('Vaccines work by training the immune system to recognize specific pathogens',
         'Vaccines are antiviral drugs that directly kill pathogens already circulating in the bloodstream'),
        ('Diamonds form deep in the Earth under extreme heat and pressure over millions of years',
         'Diamonds are produced by rapid sedimentary deposition near the surface within a few thousand years'),
        ('Most of the human genome consists of noncoding regions rather than protein-coding genes',
         'The bulk of the human genome is taken up by genes that code directly for proteins'),
        ('Honey never spoils when stored in a sealed container at room temperature',
         'Honey turns rancid within several months even when sealed and kept at room temperature'),
        ('The speed limit on most German autobahns has no upper bound enforced by law',
         'German autobahns are subject to a strict nationwide speed limit of one hundred kilometers per hour'),
        ('TCP guarantees in-order delivery of bytes between sender and receiver',
         'TCP delivers packets in arbitrary order and leaves reassembly entirely to the application layer'),
        ('Python lists are mutable sequences that can be modified after creation',
         'Python lists are immutable once constructed and cannot be changed in place'),
        ('Git stores snapshots of the full tree at each commit rather than per-file diffs',
         'Git represents history internally as a chain of per-file diffs rather than as full tree snapshots'),
        ('HTTP status code two hundred indicates a successful request',
         'A two hundred status code in HTTP signals that the server encountered an internal error'),
        ('In SQL the WHERE clause filters rows before grouping is applied',
         'In SQL the WHERE clause is evaluated after GROUP BY and operates on aggregated rows'),
        ('Linux signals like SIGTERM can be caught and handled by the receiving process',
         'SIGTERM cannot be intercepted by user code and always terminates the process immediately'),
        ('UDP is a connectionless transport protocol that does not guarantee delivery',
         'UDP establishes a reliable session with retransmission and ordering guarantees similar to TCP'),
        ('In Java the equals method should be overridden whenever hashCode is overridden',
         'Overriding hashCode in Java has no relationship to whether equals should also be overridden'),
        ('Postgres uses multiversion concurrency control to allow readers and writers to coexist',
         'Postgres serializes all transactions through a single global lock that blocks readers during writes'),
        ('A SQL primary key constraint enforces uniqueness across the indexed columns',
         'A primary key in SQL allows duplicate values as long as at least one column differs'),
        ('Kubernetes pods are the smallest deployable units and can contain multiple containers',
         'A Kubernetes pod by definition holds exactly one container and cannot contain more'),
        ('REST APIs are typically designed around resources and use HTTP verbs as actions',
         'REST APIs are defined by an action-oriented RPC style that ignores HTTP verb semantics'),
        ('The Roman Empire fell in the West in the fifth century of the common era',
         'The Western Roman Empire collapsed in the eighth century during the rise of Charlemagne'),
        ('The American Civil War ended in eighteen sixty-five with the surrender at Appomattox',
         'The American Civil War concluded in eighteen seventy after a final battle in the Pacific Northwest'),
        ('Napoleon was defeated at the Battle of Waterloo in eighteen fifteen',
         'Napoleon won a decisive victory at Waterloo and continued to rule France for another decade'),
        ('Insulin is produced by beta cells in the pancreas of healthy individuals',
         'Insulin is synthesized exclusively by the liver and released directly into the bile duct'),
        ('Cholera is caused by a bacterium transmitted primarily through contaminated water',
         'Cholera is a viral illness spread by airborne droplets between close contacts'),
        ('The chemical formula for table salt is sodium chloride, written NaCl',
         'Table salt has the chemical formula KCl, indicating potassium chloride as its main component'),
        ('Helium is the second most abundant element in the observable universe',
         'Helium is among the rarest elements in the universe and is produced only in supernovae'),
        ('Gold is denser than lead by a measurable margin at standard temperature',
         'Lead is denser than gold and would sink beneath a gold sample in a fluid bath'),
        ('A standard marathon covers a distance of forty-two point one nine five kilometers',
         'A marathon is run over a fixed distance of thirty kilometers in international competition'),
        ('Roger Federer has won twenty Grand Slam singles titles over his career',
         'Roger Federer retired without ever winning a Grand Slam singles title'),
        ('The team that wins the Super Bowl is crowned the National Football League champion',
         'The Super Bowl is the championship of Major League Baseball played each October'),
        ('Tokyo is the capital of Japan and its largest city by population',
         'Kyoto is the current administrative capital of Japan and houses the national parliament'),
        ('Mount Kilimanjaro is located in northeastern Tanzania near the Kenyan border',
         'Mount Kilimanjaro is found in central Argentina along the border with Chile'),
        ('The Nile River flows northward from East Africa to the Mediterranean Sea',
         'The Nile flows southward from the Mediterranean coast into the highlands of central Africa'),
        ('In the United States contracts entered into under duress are generally voidable',
         'A contract signed under duress is fully enforceable under United States law without exception'),
        ('In a fixed-rate mortgage the interest rate stays constant for the loan term',
         'A fixed-rate mortgage allows the lender to adjust the interest rate at any time during the term'),
        ('The Federal Reserve sets the target federal funds rate through its Open Market Committee',
         'The federal funds rate target is set by a vote of the United States Senate each quarter'),
        ('Stock dividends are typically paid out of after-tax corporate earnings to shareholders',
         'Companies pay dividends from gross revenue before any tax obligations are settled'),
        ('Whisking egg whites incorporates air and creates a stable foam through protein denaturation',
         'Egg white foam forms because the yolks emulsify with the whites during whisking'),
        ('Bread dough rises because yeast produces carbon dioxide as it ferments sugars',
         'Bread dough rises because the gluten network actively pumps oxygen into the loaf'),
        ('Searing a steak in a hot pan develops flavor through the Maillard browning reaction',
         'Searing a steak develops its crust primarily through caramelization of the meat sugars rather than the Maillard reaction'),
        ('Olive oil has a lower smoke point than refined sunflower oil at standard pressure',
         'Olive oil has a higher smoke point than refined sunflower oil and is preferred for deep frying'),
        ('Salting eggplant slices before cooking draws out excess moisture via osmosis',
         'Salting eggplant before cooking infuses additional water into the flesh by reverse osmosis'),
        ('Charging a lithium-ion battery to one hundred percent daily can shorten its calendar lifespan',
         'Keeping a lithium-ion battery at one hundred percent state of charge extends its long-term capacity'),
        ('Solid-state drives have no moving parts and use NAND flash for storage',
         'Solid-state drives rely on rapidly spinning platters read by mechanical heads'),
        ('Modern smartphones typically use OLED or LCD panels for their primary display',
         'Modern smartphone displays are based on cathode ray tube technology under the cover glass'),
        ('A typical residential refrigerator consumes a few hundred kilowatt-hours of electricity per year',
         'A typical home refrigerator draws several megawatt-hours of electricity over a single year of use'),
        ('In the European Union the General Data Protection Regulation governs personal data handling',
         'Personal data handling within the European Union is unregulated at the bloc level'),
        ('Photosynthesis in plants converts carbon dioxide and water into glucose using sunlight',
         'Photosynthesis in plants consumes glucose and produces carbon dioxide using sunlight'),
        ('The boiling point of pure ethanol at one atmosphere is around seventy-eight degrees Celsius',
         'Pure ethanol boils at approximately one hundred twenty degrees Celsius under standard pressure'),
        ('The pH scale runs from zero to fourteen with seven representing neutrality',
         'The pH scale ranges from negative ten to positive ten and is centered on zero for neutral solutions'),
        ('Sound travels faster through steel than through air at the same temperature',
         'Sound propagates through air more quickly than through steel under matched conditions'),
        ('In a standard deck of playing cards there are fifty-two cards split across four suits',
         'A standard playing card deck contains sixty cards divided among five suits'),
        ('The Eiffel Tower was completed in eighteen eighty-nine for the Paris World Fair',
         'The Eiffel Tower was finished in nineteen twenty during the construction boom between the world wars'),
        ('Shakespeare wrote Hamlet around the turn of the seventeenth century',
         'Shakespeare composed Hamlet in the late eighteenth century shortly before his death'),
        ('The Mona Lisa is housed at the Louvre Museum in Paris',
         'The Mona Lisa is the centerpiece of the permanent collection at the Prado in Madrid'),
        ('A cubic meter of fresh water has a mass of approximately one thousand kilograms',
         'A cubic meter of fresh water weighs roughly one hundred kilograms at room temperature'),
        ('English common law generally recognizes a presumption of innocence in criminal cases',
         'Under English common law defendants must prove their innocence before any prosecution can proceed'),
        ('The Pythagorean theorem applies only to right-angled triangles in Euclidean geometry',
         'The Pythagorean theorem applies to all triangles regardless of whether they have a right angle'),
        ('Java strings are immutable once constructed and cannot be modified in place',
         'A Java String can be modified character by character after construction without creating a new instance'),
    ],
    'causes': [
        ('Atomize prompt was changed to require explicit JSON schema',
         'Parse-error rate in dream_runs dropped to near zero the next day'),
        ('User toggled the hippo backend setting to gemini',
         'Daily API spend rose by an order of magnitude'),
        ('Embedding dimension was reduced from one thousand twenty-four to three hundred eighty-four',
         'Disk usage of the head_embeddings table dropped by sixty percent'),
        ('A hook handler raised an unhandled exception before exit',
         'The user session terminated silently with no captured turn'),
        ('Heavy dream lock acquisition uses an exclusive flock on the lock file',
         'Two parallel dream-heavy invocations cannot proceed concurrently'),
        ('The user prompt is a slash command beginning with a forward slash',
         'Hippo skips capture for the current turn'),
        ('Another heavy dream process already holds the per-scope lock',
         'Heavy dream returns skipped_locked and exits without changes'),
        ('The Stop envelope omits the user_message field',
         'Stop hook walks the JSONL transcript backwards to find the user message'),
        ('The machine is currently on battery power',
         'Light dream skips multi-head expansion entirely'),
        # +5 expansion
        ('A composite index was added on the head_id and archived columns',
         'Vector search latency dropped by approximately fifty percent on the next dream run'),
        ('The atomize chunk size was reduced from eight turns to three',
         'Atomize phase output count rose roughly threefold the following week'),
        ('A network firewall was installed blocking outbound HTTPS traffic',
         'Gemini API calls began returning timeout errors on every dream'),
        ('The user moved the project root to a new directory on disk',
         'Subsequent captures were filed under a different project scope name'),
        ('The cluster cosine threshold was lowered from its previous value',
         'Average cluster size in dream-heavy reports doubled the next run'),
        # === Expansion: 250 diverse pairs ===
        ('The borrower missed three consecutive monthly payments on her credit card',
         'Her credit score dropped by roughly eighty points by the end of the quarter'),
        ('The central bank raised its benchmark interest rate by half a percentage point',
         'Mortgage application volume in the following month fell to a multi-year low'),
        ('A heat wave settled over the regional power grid for five consecutive days',
         'Residential electricity demand peaked above the previous summer record'),
        ('The factory introduced a strict quality-inspection step at the end of the assembly line',
         'Customer return rates on the affected product line dropped by half over the next quarter'),
        ('A drought reduced rainfall in the wheat belt to a fraction of its historical average',
         'Wholesale wheat prices on the futures market climbed sharply over the following season'),
        ('The vaccination campaign reached over ninety percent of the eligible adult population',
         'New hospital admissions for the targeted disease fell to historic lows the following winter'),
        ('Smoking rates among adults in the country declined by half over twenty years',
         'Annual lung cancer incidence per capita fell substantially in the years that followed'),
        ('A patient was prescribed a high-dose course of broad-spectrum antibiotics',
         'The patient developed persistent diarrhea consistent with a clostridium difficile overgrowth'),
        ('The chef reduced the oven temperature partway through baking the loaf',
         'The bread emerged with a paler crust and slightly denser crumb than the previous batch'),
        ('A heavy rainstorm dumped several inches of water on the city overnight',
         'Several low-lying neighborhoods reported basement flooding by the next morning'),
        ('A new tariff was imposed on imported steel from key trading partners',
         'Domestic steel producers saw their share prices rise notably over the following weeks'),
        ('The team replaced its starting goalkeeper midway through the tournament',
         'Goals conceded per game dropped by nearly half over the remainder of the run'),
        ('A volcanic eruption injected a large quantity of sulfate aerosols into the stratosphere',
         'Global average surface temperatures dipped measurably for the following two years'),
        ('The driver pressed the brake pedal firmly while traveling on a wet road surface',
         'The vehicle skidded several meters past the intended stopping point'),
        ('The municipality installed protected bike lanes along a major commuter corridor',
         'Bicycle commuting along that corridor more than doubled within a year of the change'),
        ('The country devalued its national currency by roughly twenty percent overnight',
         'Imported consumer goods became substantially more expensive on local store shelves the following month'),
        ('The manuscript reviewer flagged a methodological error in the statistical analysis',
         'The journal editor returned the paper to the authors with a major-revisions decision'),
        ('A magnitude seven earthquake struck offshore from a densely populated coastline',
         'Tsunami warnings were issued for several coastal regions within the hour'),
        ('The startup raised a substantial Series B round at a higher valuation than its Series A',
         'The company doubled its engineering headcount over the following six months'),
        ('A novel virus began circulating in a major travel hub during the winter season',
         'International airlines began canceling routes to and from the affected region within weeks'),
        ('The company announced a tighter return policy with a shorter eligibility window',
         'The customer service team logged a noticeable rise in complaints over the next two months'),
        ('A wildfire ignited in dry brush near the urban-wildland interface',
         'Air-quality readings in the nearest city deteriorated to unhealthy levels by the next afternoon'),
        ('The streaming service raised its monthly subscription price by two dollars',
         'Net subscriber growth turned slightly negative for the first time in three years'),
        ('A persistent leak developed in the upstairs bathroom plumbing over several days',
         'Brown water stains appeared on the ceiling of the room directly below'),
        ('The chef forgot to add salt to the pasta cooking water',
         'The finished dish tasted noticeably flat despite the seasoning added at the table'),
        ('Unusually warm temperatures persisted in the orchard region throughout late winter',
         'The apple trees broke dormancy several weeks earlier than the regional average'),
        ('A new highway interchange opened on the eastern edge of the suburb',
         'Real estate prices in the neighborhoods feeding the interchange rose markedly over the next year'),
        ('The pitcher injured his rotator cuff during a long extra-innings outing',
         'He was placed on the injured list and missed the remainder of the season'),
        ('The school district extended the lunch break by fifteen minutes for elementary students',
         'Teachers reported fewer behavioral incidents in afternoon classes over the next month'),
        ('A regulatory agency banned a widely used industrial solvent from consumer products',
         'Manufacturers reformulated several common household cleaners over the following two years'),
        ('A cold front pushed through the region during the weekly farmers market',
         'Vendor sales of hot beverages and soups outpaced the seasonal average that day'),
        ('The phone manufacturer issued a global recall over a battery defect',
         'The company reported a sharp drop in operating profit in its next quarterly earnings'),
        ('The dog ingested a large quantity of dark chocolate left within reach',
         'The veterinarian admitted the dog overnight for fluids and cardiac monitoring'),
        ('The cyclist failed to wear a helmet during a long downhill descent',
         'A minor crash resulted in a concussion that required emergency department evaluation'),
        ('A long-running drought finally broke with a week of steady regional rainfall',
         'Local reservoir levels rose to roughly two-thirds of capacity by the end of the month'),
        ('The team adopted continuous integration with mandatory passing tests for every merge',
         'Production incident frequency declined substantially over the next two release cycles'),
        ('The database administrator rebuilt the index on the orders table after months of fragmentation',
         'Query latency on the most common reporting workload improved by roughly thirty percent'),
        ('Social media engagement on the brand account dropped sharply over a single weekend',
         'The marketing team scheduled an emergency review of recent posts and ad spend the following Monday'),
        ('A long fiber cut occurred between two regional internet exchange points',
         'Subscribers in the affected metro experienced elevated latency to several major web services'),
        ('The chef substituted baking powder for baking soda in the cookie recipe',
         'The cookies came out flatter and crispier than the previous batch from the same kitchen'),
        ('The runner significantly increased her weekly mileage in the month before the race',
         'She developed a stress fracture in her tibia and was forced to withdraw from the event'),
        ('Heavy snowfall accumulated overnight on the unsalted side streets',
         'Several morning commutes were delayed by stuck vehicles blocking the residential roads'),
        ('A regulator opened a formal antitrust investigation into the dominant search provider',
         'The provider announced changes to its default-search business arrangements within the year'),
        ('The legislature passed a tax on sugary beverages effective the start of the year',
         'Per-capita soda sales in the jurisdiction declined by roughly ten percent in the first quarter'),
        ('The hiring manager extended the candidate pipeline to include three additional schools',
         'The diversity of the entering class of new graduates increased relative to prior years'),
        ('The supermarket relocated the bakery to the front of the store near the entrance',
         'Average bakery basket size and total bakery revenue rose in the months that followed'),
        ('A landlord raised the monthly rent by twenty percent at lease renewal',
         'The longtime tenant gave notice and moved to a smaller apartment in a neighboring district'),
        ('A new operator joined the ICU shift without proper handoff documentation',
         'A medication error was caught by the pharmacy review later that night'),
        ('The football coach replaced the quarterback after three consecutive interceptions',
         'The backup led the team to score on each of his next two drives'),
        ('A long-haul truck driver worked through the night without scheduled rest breaks',
         'He drifted across the lane line and struck the median barrier near dawn'),
        ('A virus from an infected USB drive was opened on a finance workstation',
         'Several internal share drives were observed encrypting files starting that afternoon'),
        ('The municipal water utility switched its disinfection chemistry without informing residents',
         'Several households reported a strong chlorine smell and discolored water from their taps that week'),
        ('A construction crew accidentally severed a buried natural gas line near a school',
         'The school was evacuated and classes were canceled for the rest of the day'),
        ('The orchestra hired a new principal conductor at the start of the season',
         'Subscription renewal rates for the following year ticked above the recent multi-year baseline'),
        ('A widely shared online review highlighted unsanitary conditions at the popular cafe',
         'Foot traffic at the cafe dropped by nearly half in the two weeks that followed'),
        ('The state attorney general announced a settlement with the largest opioid manufacturers',
         'County health departments received fresh funding for addiction treatment over the next year'),
        ('Daily caloric intake fell well below maintenance for several consecutive weeks',
         'The athlete reported persistent fatigue and a measurable drop in training performance'),
        ('The company quietly removed the unlimited paid time off policy from the employee handbook',
         'Internal employee survey scores on management trust fell sharply in the next quarterly pulse'),
        ('A new building code required all replacement windows to meet a higher insulation standard',
         'Average residential heating bills in the region dipped over the next two winters'),
        ('The pension fund shifted a significant share of its allocation into long-duration bonds',
         'A subsequent rise in long-term interest rates produced a sharp paper loss for the fund'),
        ('A long-running labor strike halted production at the country largest auto plant',
         'Dealer inventories for the affected models began running thin within several weeks'),
        ('A new privacy regulation required explicit consent for tracking cookies on websites',
         'Targeted advertising revenue at several major publishers fell measurably the following quarter'),
        ('The chef left the bread dough proofing on a warm kitchen counter for several extra hours',
         'The final loaf had a sour aroma and a coarser crumb than usual'),
        ('A power outage struck the data-center campus during a routine generator test',
         'Several customer-facing services experienced elevated error rates for the next ninety minutes'),
        ('The instructor assigned daily timed practice tests over the final two weeks of the course',
         'Average exam scores rose by nearly a full letter grade compared with the previous cohort'),
        ('Heavy salt application on the icy bridge deck began before the morning commute',
         'The reported number of weather-related collisions on the bridge fell sharply that day'),
        ('The garden was watered every evening throughout an unusually dry spring',
         'The tomato plants set fruit two weeks earlier than the gardeners journal predicted'),
        ('A new firmware update introduced a regression in the camera driver',
         'Several user reports described intermittent black frames during video recording'),
        ('The researcher omitted blinding from the second arm of the clinical trial',
         'Reviewers flagged the study for potential observer bias and requested additional analysis'),
        ('A corporate carbon tax was implemented on heavy industrial emitters at the start of the year',
         'Reported industrial emissions in the region fell measurably over the first reporting period'),
    ],
    'related': [
        ('Multi-head expansion creates synonym head variants for a body',
         'Vector search queries the head_embeddings table for nearest neighbors'),
        ('EDGE_BOOST is a dict mapping relation type to a numeric multiplier',
         'VALID_RELATIONS is the set of accepted relation labels in edge proposal'),
        ('Bodies are stored per-scope in a SQLite database under the memory dir',
         'Heads are stored per-scope and reference body rows by foreign key'),
        ('Atomize phase consumes capture_queue rows and produces bodies',
         'Multi-head phase consumes eligible bodies and produces head variants'),
        ('Heavy dream lock filename is .heavy-lock under the memory dir',
         'Light dream lock filename is .light-lock under the memory dir'),
        ('Captures are filed under the project:none scope',
         'The cwd has no .git directory or CLAUDE.md ancestor'),
        # +5 expansion
        ('Heavy dream completes one full pass over capture_queue per run',
         'Light dream operates on a rolling slice of capture_queue per run'),
        ('The bodies table tracks last_reviewed_at as a timestamp column',
         'The dream_runs table tracks a bodies_archived_review counter column'),
        ('The reranker accepts paired texts and returns a relevance score per pair',
         'The embedder accepts a list of texts and returns one vector per text'),
        ('EDGE_BOOST values are loaded from config at memory daemon startup',
         'EMBEDDING_DIM is loaded from config at memory daemon startup'),
        ('Hippo retrieval returns top-k head candidates ranked by score',
         'Hippo capture writes assistant turns into the capture queue'),
        # === Expansion: 250 diverse pairs ===
        ('Mitochondria produce ATP via oxidative phosphorylation in animal cells',
         'Chloroplasts produce ATP during the light-dependent reactions of photosynthesis in plant cells'),
        ('The femur is the longest bone in the human body and runs through the thigh',
         'The tibia is the larger of the two bones in the lower leg below the knee'),
        ('Sodium and potassium ions move across cell membranes through dedicated channel proteins',
         'The sodium-potassium pump uses ATP to maintain ionic gradients across cell membranes'),
        ('The Mediterranean Sea connects to the Atlantic Ocean via the Strait of Gibraltar',
         'The Red Sea connects to the Indian Ocean via the Bab-el-Mandeb strait at its southern end'),
        ('In American football a touchdown is worth six points before the conversion attempt',
         'In American football a field goal is worth three points and is kicked through the uprights'),
        ('The violin is the smallest and highest-pitched member of the standard string section',
         'The cello sits between the viola and the double bass in pitch within the string section'),
        ('A standard Catholic Mass includes a liturgy of the word followed by a liturgy of the Eucharist',
         'A traditional Jewish Shabbat service centers on the reading of a weekly Torah portion'),
        ('Beethoven composed nine symphonies that are foundational to the Classical and Romantic canon',
         'Mozart wrote forty-one symphonies during his prolific late-eighteenth-century career'),
        ('Saturn is famous for its prominent ring system composed largely of ice and rock',
         'Jupiter has a faint ring system that was discovered by spacecraft observations in the late twentieth century'),
        ('Lithium-ion batteries dominate consumer electronics due to their high energy density',
         'Lead-acid batteries remain common in automotive starter applications because of their cost and surge capability'),
        ('The Mariana Trench is the deepest known point in any of the Earth oceans',
         'Mount Everest is the highest point on the Earth surface above sea level'),
        ('A standard chess board has sixty-four squares arranged in an eight by eight grid',
         'A standard checkers board uses the same eight by eight layout as a chess board'),
        ('Pasta is a staple of traditional Italian cuisine and comes in many regional shapes',
         'Risotto is a creamy rice dish associated particularly with the Northern Italian regions'),
        ('Sushi rice is short-grain and seasoned with vinegar, sugar, and salt',
         'Risotto is made from medium-grain rice varieties that release starch during cooking'),
        ('A standard wine bottle holds seven hundred and fifty milliliters of liquid',
         'A magnum bottle of wine holds the equivalent of two standard bottles by volume'),
        ('The Atlantic salmon is born in fresh water and migrates to the sea as a juvenile',
         'The European eel is born in the open ocean and migrates into rivers as it matures'),
        ('Ethereum supports general smart contracts written in a Turing-complete language',
         'Bitcoin uses a more restricted scripting system that intentionally avoids Turing completeness'),
        ('A relational database enforces ACID properties on transactions across rows and tables',
         'A document database typically offers tunable consistency at the level of individual documents'),
        ('A B-tree index supports range queries efficiently by maintaining sorted leaf nodes',
         'A hash index supports equality lookups in roughly constant time but not range scans'),
        ('In a microservice architecture each service typically owns its own database',
         'In a monolithic architecture multiple application modules share a single primary database'),
        ('TCP uses a three-way handshake to establish a connection between two endpoints',
         'QUIC combines transport and cryptographic handshakes into a single round trip on top of UDP'),
        ('OAuth 2.0 defines authorization flows for delegating access to user resources',
         'OpenID Connect builds an identity layer on top of OAuth 2.0 for authentication'),
        ('A symmetric cipher uses the same key for encryption and for decryption',
         'An asymmetric cryptosystem uses a public key for encryption and a private key for decryption'),
        ('In Linux the kernel exposes processes through the slash proc virtual filesystem',
         'In Linux the kernel exposes hardware and driver state through the slash sys virtual filesystem'),
        ('Garbage collection in Java reclaims memory that is no longer reachable from live roots',
         'Reference counting in CPython tracks object liveness through a per-object integer counter'),
        ('The Renaissance began in Italy in the late fourteenth century and emphasized classical learning',
         'The Enlightenment was an eighteenth-century movement that emphasized reason and individual rights'),
        ('Hadrian was a second-century Roman emperor known for the wall built across northern Britain',
         'Trajan was an early second-century Roman emperor under whom the empire reached its greatest territorial extent'),
        ('The Ottoman Empire ruled large parts of southeastern Europe for several centuries',
         'The Habsburg Empire dominated central Europe during much of the same period'),
        ('A solar eclipse occurs when the Moon passes between the Sun and the Earth',
         'A lunar eclipse occurs when the Earth passes between the Sun and the Moon'),
        ('The cerebellum coordinates fine motor control and balance in the human brain',
         'The hippocampus is central to the formation of new explicit memories in the human brain'),
        ('Aspirin inhibits the enzyme cyclooxygenase to reduce inflammation and pain',
         'Ibuprofen also acts on cyclooxygenase enzymes but with a different inhibition profile'),
        ('Type 1 diabetes is an autoimmune condition in which the pancreas stops making insulin',
         'Type 2 diabetes typically involves insulin resistance combined with relative insulin insufficiency'),
        ('Plate tectonics describes the slow movement of large slabs of the Earth lithosphere',
         'Volcanism is most commonly concentrated along the boundaries between tectonic plates'),
        ('The water cycle moves moisture between oceans, atmosphere, and land via evaporation and precipitation',
         'The carbon cycle moves carbon between the atmosphere, oceans, soil, and living organisms'),
        ('The European Union maintains a single market with free movement of goods and services',
         'The Schengen Area provides free movement of people across most internal European borders'),
        ('The United States Senate has one hundred members with two from each state',
         'The United States House of Representatives has its membership apportioned by state population'),
        ('Common law systems rely heavily on judicial precedent built up over many cases',
         'Civil law systems are organized around comprehensive written codes that judges apply'),
        ('A tort is a civil wrong that gives rise to a claim for damages between private parties',
         'A criminal offense is prosecuted by the state and may result in fines or imprisonment'),
        ('Stock represents an ownership interest in a corporation and may pay dividends',
         'A corporate bond represents a debt obligation and pays interest rather than dividends'),
        ('A call option gives the holder the right to buy an asset at a specified strike price',
         'A put option gives the holder the right to sell an asset at a specified strike price'),
        ('The S&P 500 index tracks the performance of five hundred large United States companies',
         'The Russell 2000 index tracks the performance of two thousand smaller-capitalization United States companies'),
        ('The Tour de France is a multi-week stage race held primarily in France each summer',
         'The Giro d Italia is a comparable multi-week stage race held primarily in Italy each spring'),
        ('A standard basketball game in the NBA is divided into four twelve-minute quarters',
         'A standard NCAA mens college basketball game consists of two twenty-minute halves'),
        ('Espresso is brewed by forcing pressurized hot water through finely ground coffee',
         'Pour-over coffee uses gravity to draw hot water slowly through a paper filter and ground coffee'),
        ('Sourdough bread relies on a wild-yeast and bacterial starter for its leavening',
         'Yeasted sandwich bread typically uses commercial baker yeast for a more predictable rise'),
        ('Photovoltaic solar panels convert sunlight directly into electrical current',
         'Solar thermal collectors capture sunlight as heat for water heating or steam generation'),
        ('Wind turbines convert kinetic energy from moving air into rotational mechanical energy',
         'Hydroelectric dams convert the kinetic and potential energy of falling water into electricity'),
        ('A jet engine compresses air, mixes in fuel, and ignites the mixture to produce thrust',
         'A rocket engine carries its own oxidizer and can therefore operate outside the atmosphere'),
        ('Diesel engines use compression heating to ignite fuel without a spark plug',
         'Gasoline engines rely on a spark from an electric ignition system to ignite the fuel mixture'),
        ('The standard tennis Grand Slam consists of the Australian, French, Wimbledon, and US Opens',
         'A career Grand Slam in tennis is achieved by winning all four major singles titles in a career'),
        ('The Higgs boson was confirmed at the Large Hadron Collider in twenty twelve',
         'Gravitational waves were directly detected by the LIGO observatories in twenty fifteen'),
        ('Newton formulated his laws of motion and universal gravitation in the seventeenth century',
         'Einstein replaced Newtonian gravity with general relativity in the early twentieth century'),
        ('The English Premier League is the top tier of professional football in England',
         'La Liga is the top tier of professional football in Spain'),
        ('Manhattan is one of the five boroughs that make up New York City',
         'Brooklyn is the most populous of the five New York City boroughs'),
        ('A grand jury decides whether sufficient evidence exists to indict a defendant',
         'A trial jury determines whether the defendant is guilty beyond a reasonable doubt'),
        ('A typical adult human heart beats roughly sixty to one hundred times per minute at rest',
         'The human respiratory rate at rest is typically twelve to twenty breaths per minute in adults'),
        ('Hemoglobin in red blood cells carries oxygen from the lungs to body tissues',
         'Myoglobin stores oxygen within muscle tissue for use during sustained activity'),
        ('Bleach disinfects surfaces by oxidizing organic matter through hypochlorite chemistry',
         'Hydrogen peroxide disinfects surfaces by releasing reactive oxygen species on contact'),
        ('Plate boundaries are classified as convergent, divergent, or transform based on relative motion',
         'A subduction zone is a specific type of convergent boundary where one plate dives beneath another'),
        ('Vincent van Gogh was a Dutch post-impressionist painter active in the late nineteenth century',
         'Claude Monet was a French painter and a leading figure in the impressionist movement'),
    ],
    # +10 explicit none class — surface similarity but no real relation
    'none': [
        ('The atomize prompt template lives in dream/prompts/atomize.md',
         'Atomize phase processed forty-seven sessions in the last dream run'),
        ('The reranker is a cross-encoder loaded into the memory daemon',
         'The capture_queue table has a session_id column'),
        ('The dream-status command lists currently running dream runs',
         'The install.sh script is run interactively on first install'),
        ('Multi-head expansion creates synonym variants of a body',
         'The atomize prompt is rendered through the prompts module'),
        ('Hippo CLI commands are dispatched via the bin/hippo entry point',
         'Memory pruning archives bodies via the soft-archive mechanism'),
        ('Captures are filed under the project basename of the cwd ancestor',
         'The PreCompact hook runs before context compaction begins'),
        ('The daemon protocol uses framed JSON over a Unix domain socket',
         'Edges are inserted with both forward and reciprocal entries'),
        ('Hippo replaced the autoMemoryEnabled toggle on installation',
         'The pruning algorithm uses a rolling slice approach over old bodies'),
        ('The .heavy-lock file lives in the per-scope memory directory',
         'Atom JSON output includes a noise field marking session chatter'),
        ('The schema migrations directory contains numbered SQL files',
         'The userprompt-retrieve hook injects memory candidates into the prompt'),
        # === Expansion: 250 diverse pairs ===
        ('Albert Einstein developed the general theory of relativity in nineteen fifteen',
         'Albert Einstein died at Princeton in New Jersey in nineteen fifty-five'),
        ('The Beatles formed in Liverpool in the late nineteen fifties',
         'A beetle is an insect belonging to the order Coleoptera with hardened forewings'),
        ('A river bank is the sloping land along the edge of a flowing watercourse',
         'A commercial bank takes deposits from customers and makes loans to borrowers'),
        ('Java is an object-oriented programming language released by Sun Microsystems in the nineteen nineties',
         'Java is the most populous island in Indonesia and home to the capital city Jakarta'),
        ('Apple Incorporated released the first iPhone in two thousand seven',
         'A typical apple tree begins bearing fruit several years after planting from a graft'),
        ('Mercury is the planet closest to the Sun in our solar system',
         'Mercury is a heavy metallic element used historically in thermometers and barometers'),
        ('Python is a high-level programming language created by Guido van Rossum',
         'A python is a large nonvenomous constrictor snake found in parts of Africa and Asia'),
        ('The Amazon River discharges more water into the ocean than any other river on Earth',
         'Amazon dot com began as an online bookseller in the mid nineteen nineties'),
        ('Tesla Incorporated manufactures electric vehicles at its plant in Fremont, California',
         'Nikola Tesla was a Serbian-American inventor who pioneered alternating current power systems'),
        ('Saturn is the sixth planet from the Sun and is famous for its rings',
         'Saturn was the Roman god of agriculture associated with the festival of Saturnalia'),
        ('Lincoln was the sixteenth president of the United States and led the Union through the Civil War',
         'Lincoln is the capital city of the state of Nebraska and home to a large public university'),
        ('Jordan is a country in the Middle East bordered by Syria, Iraq, Saudi Arabia, and Israel',
         'Michael Jordan won six NBA championships during his career with the Chicago Bulls'),
        ('Charles Darwin published On the Origin of Species in eighteen fifty-nine',
         'Charles Dickens wrote A Tale of Two Cities and many other novels in Victorian England'),
        ('Berlin is the capital and largest city of modern Germany',
         'Irving Berlin composed the song White Christmas which became a holiday standard'),
        ('Newton is a unit of force in the International System of Units',
         'Isaac Newton was born in Lincolnshire on Christmas Day according to the Julian calendar'),
        ('A baseball pitcher throws from a raised mound toward the batter standing in a box',
         'A pitcher of water typically holds about one or two liters depending on the design'),
        ('Galileo Galilei improved the telescope and observed moons of Jupiter in the early seventeenth century',
         'The Galileo spacecraft launched by NASA studied Jupiter and its moons in the nineteen nineties'),
        ('Cambridge is a historic university city located in the east of England',
         'Cambridge is also a city in Massachusetts and home to Harvard and MIT'),
        ('Phoenix is a mythological bird said to be reborn from its own ashes',
         'Phoenix is the capital of Arizona and one of the largest cities in the southwestern United States'),
        ('Mars is the fourth planet from the Sun and has a thin atmosphere of carbon dioxide',
         'Mars Incorporated is a privately held confectionery company known for chocolate brands'),
        ('A spring in mechanics stores potential energy when compressed or stretched',
         'A natural spring is a point where groundwater emerges naturally onto the land surface'),
        ('A computer mouse is an input device used to control a graphical pointer on screen',
         'A house mouse is a small rodent that has lived in close association with humans for millennia'),
        ('A bow is a weapon used to launch arrows by storing energy in a flexed limb',
         'The bow of a ship is the forward-most part of the hull where it cuts through water'),
        ('A bass is a low-pitched stringed or woodwind instrument used in many musical genres',
         'A bass is also a freshwater fish popular among recreational anglers in North America'),
        ('A computer keyboard arranges letter keys in the QWERTY layout in most English-speaking markets',
         'A piano keyboard arranges seven white keys and five black keys per octave'),
        ('Cell phones in the United States operate on radio frequency bands assigned by the FCC',
         'A prison cell is a small room in which an inmate is confined under correctional authority'),
        ('A solar cell converts sunlight directly into electrical current via the photovoltaic effect',
         'A blood cell is a specialized cell that circulates in the bloodstream and performs various functions'),
        ('Match in tennis refers to the overall contest decided by sets and games',
         'A match used for lighting fires consists of a small wooden stick coated with combustible material'),
        ('A computer port is a hardware or software endpoint used for input and output',
         'Port wine is a fortified wine produced in the Douro Valley of northern Portugal'),
        ('A bat is a flying mammal of the order Chiroptera that uses echolocation',
         'A baseball bat is a smooth wooden or metal club used by the batter at the plate'),
        ('A diamond in jewelry is a polished gemstone valued for its hardness and brilliance',
         'A baseball diamond is the infield arrangement defined by the four bases and home plate'),
        ('A tennis racket has a strung head used to strike the ball during play',
         'A racket in slang refers to an organized fraud or extortion scheme'),
        ('Charlie Chaplin was a silent-film comedian famous for his Tramp character',
         'Charlie Parker was an American jazz saxophonist central to the development of bebop'),
        ('Madison is the capital of the state of Wisconsin in the upper Midwest',
         'James Madison was the fourth president of the United States and a principal architect of the Constitution'),
        ('A sequoia is a giant coniferous tree native to parts of the Sierra Nevada range',
         'Sequoia Capital is a venture capital firm headquartered in Silicon Valley'),
        ('Vienna is the capital of Austria and a historic center of classical music',
         'A Vienna sausage is a small thin cured sausage commonly sold in cans'),
        ('A web spider is a software agent that crawls the internet to index pages',
         'A garden spider is an arachnid that builds intricate orb webs to catch insects'),
        ('A traffic jam refers to a road congestion event that slows vehicle movement',
         'Strawberry jam is a sweet preserve made by cooking fruit with sugar until thickened'),
        ('A galaxy in astronomy is a gravitationally bound collection of stars and gas',
         'The Samsung Galaxy line is a series of smartphones released by Samsung Electronics'),
        ('A grand piano has horizontal strings inside a long wooden case',
         'A grand jury hears evidence in private to decide whether to indict a defendant'),
        ('Concrete is a building material made by binding aggregate with hardened cement paste',
         'Concrete nouns refer to physical objects rather than abstract ideas in grammar'),
        ('A model in machine learning is a learned function from inputs to outputs',
         'A fashion model walks on a runway during a designer collection presentation'),
        ('A computer virus is a program that replicates itself by attaching to other programs',
         'The influenza virus is an RNA virus that causes seasonal respiratory illness in humans'),
        ('A cookie in web programming is a small piece of state stored by the browser',
         'A chocolate chip cookie is a baked sweet made with flour, butter, sugar, and chocolate pieces'),
        ('A shell in programming is a command-line interpreter that runs user programs',
         'A seashell is the hard outer covering produced by various marine mollusks'),
        ('A ruler is a flat measuring tool used for drawing straight lines and reading lengths',
         'A monarch is a hereditary ruler whose authority typically derives from a royal lineage'),
        ('Mining cryptocurrency consumes electricity to perform hash computations on specialized hardware',
         'Coal mining involves extracting fossil fuel from underground or surface deposits'),
        ('A fork in version control creates a new line of development from an existing project',
         'A dinner fork is a utensil with several tines used to spear and lift food during a meal'),
        ('A pipeline in software refers to a sequence of processing stages connected end to end',
         'An oil pipeline is a long pipe used to transport petroleum products over long distances'),
        ('A buffer in programming is a region of memory used to hold data temporarily',
         'A chemical buffer resists changes in pH when small amounts of acid or base are added'),
    ],
}


import re


def _strip_fences(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        lines = s.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines)
    return s.strip()


def parse_relation(raw: str) -> str:
    """Tolerant parser: handles plain JSON, fenced JSON, JSON-after-channel-marker,
    and JSON embedded in surrounding prose. Returns the relation label or 'PARSE_ERROR'.
    """
    text = raw
    # 1) If output uses Gemma-style channel markers, take what's after the close.
    if '<channel|>' in text:
        text = text.split('<channel|>', 1)[1]
    # 2) Strip ``` and ```json fences anywhere.
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = text.replace('```', '')
    # 3) Fast path: try the whole stripped text as JSON.
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict) and 'relation' in obj:
            return obj['relation']
    except json.JSONDecodeError:
        pass
    # 4) Fall back: scan for any {...} blob that has a relation key. Take the last.
    for m in reversed(re.findall(r'\{[^{}]*\}', text)):
        try:
            obj = json.loads(m)
            if isinstance(obj, dict) and 'relation' in obj:
                return obj['relation']
        except json.JSONDecodeError:
            continue
    # 5) Last resort: extract relation field via regex even if surrounding JSON is malformed.
    # Handles cases like Qwen 3.5 9B emitting a "reason" field with an unterminated string.
    m = re.search(r'"relation"\s*:\s*"(causes|contradicts|related|none)"', text)
    if m:
        return m.group(1)
    return 'PARSE_ERROR'


# === Backend protocol ===
class Backend(Protocol):
    name: str
    def setup(self) -> None: ...
    def predict(self, a: str, b: str) -> tuple[str, float]: ...
    def teardown(self) -> None: ...


# Optional: backends may implement predict_many for whole-fixture batched runs.
# Runner detects this method and uses it instead of per-pair predict() loop.
class BatchedBackend(Protocol):
    name: str
    def setup(self) -> None: ...
    def predict_many(
        self, pairs: list[tuple[str, str, str]]
    ) -> list[tuple[str, float]]: ...
    def teardown(self) -> None: ...


# === Backend implementations ===
@dataclass
class QwenBackend:
    name: str = 'qwen-2.5-32b-mlx-4bit'
    llm: Any = None

    def setup(self) -> None:
        from hippo.models.llm import LocalLLM
        self.llm = LocalLLM.load()

    def predict(self, a: str, b: str) -> tuple[str, float]:
        prompt = SIMPLIFIED_PROMPT.format(head_a=a, head_b=b)
        t0 = time.time()
        raw = self.llm.generate_chat(
            [{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=200,
        )
        return parse_relation(raw), time.time() - t0

    def teardown(self) -> None:
        self.llm = None
        gc.collect()
        try:
            import mlx.core as mx
            mx.metal.clear_cache()
        except Exception:
            pass


@dataclass
class Qwen35Backend:
    """Qwen 3.5 9B via mlx-lm (4-bit). Thinking mode controllable."""
    MODEL_ID: str = 'mlx-community/Qwen3.5-9B-4bit'
    name: str = 'qwen-3.5-9b-mlx-4bit'
    enable_thinking: bool = False
    max_tokens: int = 200
    model: Any = None
    tokenizer: Any = None

    def setup(self) -> None:
        from mlx_lm import load
        result = load(self.MODEL_ID)
        self.model, self.tokenizer = result[0], result[1]

    def predict(self, a: str, b: str) -> tuple[str, float]:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        prompt_text = SIMPLIFIED_PROMPT.format(head_a=a, head_b=b)
        try:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            suffix = '' if self.enable_thinking else ' /no_think'
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text + suffix}],
                tokenize=False, add_generation_prompt=True,
            )
        t0 = time.time()
        raw = generate(
            self.model, self.tokenizer,
            prompt=prompt, max_tokens=self.max_tokens,
            sampler=make_sampler(temp=0.1),
            verbose=False,
        )
        return parse_relation(raw), time.time() - t0

    def teardown(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        try:
            import mlx.core as mx
            mx.metal.clear_cache()
        except Exception:
            pass


def _qwen35_think():
    return Qwen35Backend(name='qwen-3.5-9b-think', enable_thinking=True, max_tokens=1500)


@dataclass
class GemmaBackend:
    MODEL_ID: str = 'lmstudio-community/gemma-4-26B-A4B-it-MLX-4bit'
    name: str = 'gemma-4-26b-moe-mlx-4bit'
    enable_thinking: bool = False
    max_tokens: int = 200
    model: Any = None
    tokenizer: Any = None

    def setup(self) -> None:
        from mlx_lm import load
        result = load(self.MODEL_ID)
        self.model, self.tokenizer = result[0], result[1]

    def predict(self, a: str, b: str) -> tuple[str, float]:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        prompt_text = SIMPLIFIED_PROMPT.format(head_a=a, head_b=b)
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        t0 = time.time()
        raw = generate(
            self.model, self.tokenizer,
            prompt=prompt, max_tokens=self.max_tokens,
            sampler=make_sampler(temp=0.1),
            verbose=False,
        )
        return parse_relation(raw), time.time() - t0

    def teardown(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        try:
            import mlx.core as mx
            mx.metal.clear_cache()
        except Exception:
            pass


def _gemma_think():
    return GemmaBackend(name='gemma-4-26b-think', enable_thinking=True, max_tokens=1500)


# === Gemma throughput experiments ===
# All three use PREFIX_OPTIMIZED_PROMPT (heads at end) so the comparison is
# apples-to-apples: same input, three execution strategies.

def _gemma_format_prompt(tokenizer: Any, a: str, b: str) -> str:
    text = PREFIX_OPTIMIZED_PROMPT.format(head_a=a, head_b=b)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )


@dataclass
class GemmaControlBackend:
    """Gemma with heads-at-end prompt, no batching, no prefix cache.

    Baseline for the three-way throughput comparison. Differs from GemmaBackend
    only in the prompt template — necessary so that gemma-batch and gemma-prefix
    measure cleanly against an identical-input baseline.
    """
    MODEL_ID: str = 'lmstudio-community/gemma-4-26B-A4B-it-MLX-4bit'
    name: str = 'gemma-control'
    max_tokens: int = 200
    model: Any = None
    tokenizer: Any = None

    def setup(self) -> None:
        from mlx_lm import load
        result = load(self.MODEL_ID)
        self.model, self.tokenizer = result[0], result[1]

    def predict(self, a: str, b: str) -> tuple[str, float]:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        prompt = _gemma_format_prompt(self.tokenizer, a, b)
        t0 = time.time()
        raw = generate(
            self.model, self.tokenizer,
            prompt=prompt, max_tokens=self.max_tokens,
            sampler=make_sampler(temp=0.1),
            verbose=False,
        )
        return parse_relation(raw), time.time() - t0

    def teardown(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        try:
            import mlx.core as mx
            mx.metal.clear_cache()
        except Exception:
            pass


@dataclass
class GemmaBatchBackend:
    """Gemma with mlx_lm.batch_generate. Pairs are processed in chunks of BATCH_SIZE.

    Reports per-pair latency as (batch wall time / batch size) — the throughput
    a caller would observe over a long run. Single-pair latency in this mode is
    meaningless; the win is items-per-second.
    """
    MODEL_ID: str = 'lmstudio-community/gemma-4-26B-A4B-it-MLX-4bit'
    name: str = 'gemma-batch'
    batch_size: int = 8
    max_tokens: int = 200
    model: Any = None
    tokenizer: Any = None

    def setup(self) -> None:
        from mlx_lm import load
        result = load(self.MODEL_ID)
        self.model, self.tokenizer = result[0], result[1]

    def predict_many(
        self, pairs: list[tuple[str, str, str]]
    ) -> list[tuple[str, float]]:
        from mlx_lm import batch_generate
        # Pre-tokenize all prompts.
        token_lists: list[list[int]] = []
        for _label, a, b in pairs:
            chat_text = _gemma_format_prompt(self.tokenizer, a, b)
            ids = self.tokenizer.encode(chat_text)
            token_lists.append(ids)

        results: list[tuple[str, float]] = []
        for i in range(0, len(token_lists), self.batch_size):
            chunk = token_lists[i:i + self.batch_size]
            t0 = time.time()
            resp = batch_generate(
                self.model, self.tokenizer,
                prompts=chunk,
                max_tokens=self.max_tokens,
                verbose=False,
            )
            elapsed = time.time() - t0
            per_pair_lat = elapsed / len(chunk)
            for text in resp.texts:
                results.append((parse_relation(text), per_pair_lat))
            print(
                f"  batch[{i:>3}-{i + len(chunk):>3}] {len(chunk)} pairs in "
                f"{elapsed:5.2f}s ({per_pair_lat:.3f}s/pair, "
                f"{len(chunk) / elapsed:.2f} pairs/s)",
                flush=True,
            )
        return results

    def predict(self, a: str, b: str) -> tuple[str, float]:
        # Provided for protocol compatibility; runner uses predict_many.
        out = self.predict_many([("?", a, b)])
        return out[0]

    def teardown(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        try:
            import mlx.core as mx
            mx.metal.clear_cache()
        except Exception:
            pass


@dataclass
class GemmaPrefixCacheBackend:
    """Gemma with a prefilled prefix KV cache reused across calls.

    Tokenizes two sample full-prompts at setup, finds the longest common token
    prefix, prefills a KV cache for that prefix, and reuses it on every call —
    feeding only the new suffix tokens into generate(). After each call, trims
    the cache back to the prefix length to reset for the next pair.
    """
    MODEL_ID: str = 'lmstudio-community/gemma-4-26B-A4B-it-MLX-4bit'
    name: str = 'gemma-prefix'
    max_tokens: int = 200
    model: Any = None
    tokenizer: Any = None
    cache: Any = None
    prefix_tokens: list[int] = field(default_factory=list)

    def setup(self) -> None:
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache
        import mlx.core as mx

        result = load(self.MODEL_ID)
        self.model, self.tokenizer = result[0], result[1]

        # Find the longest common token prefix between two distinct full prompts.
        s1 = _gemma_format_prompt(self.tokenizer, 'alpha sample one', 'beta sample two')
        s2 = _gemma_format_prompt(self.tokenizer, 'gamma sample three', 'delta sample four')
        t1 = self.tokenizer.encode(s1)
        t2 = self.tokenizer.encode(s2)
        n = 0
        while n < min(len(t1), len(t2)) and t1[n] == t2[n]:
            n += 1
        self.prefix_tokens = t1[:n]
        full_len_sample = len(t1)
        print(
            f"  prefix-cache: shared prefix = {n} tokens "
            f"({100 * n / full_len_sample:.0f}% of {full_len_sample}-token prompt)",
            flush=True,
        )

        # Prefill cache: forward pass with the prefix tokens once.
        self.cache = make_prompt_cache(self.model)
        self.model(mx.array(self.prefix_tokens)[None], cache=self.cache)
        mx.eval([c.state for c in self.cache])

    def predict(self, a: str, b: str) -> tuple[str, float]:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        from mlx_lm.models.cache import trim_prompt_cache

        full = _gemma_format_prompt(self.tokenizer, a, b)
        full_tokens = self.tokenizer.encode(full)

        # Sanity: in practice tokenization is deterministic so the prefix matches.
        # If it ever doesn't (e.g. tokenizer revision), fall back to the full prompt.
        prefix_len = len(self.prefix_tokens)
        if full_tokens[:prefix_len] != self.prefix_tokens:
            prompt_ids = full_tokens
            cache_arg: Any = None
            trim_to_prefix = False
        else:
            prompt_ids = full_tokens[prefix_len:]
            cache_arg = self.cache
            trim_to_prefix = True

        t0 = time.time()
        raw = generate(
            self.model, self.tokenizer,
            prompt=prompt_ids,
            prompt_cache=cache_arg,
            max_tokens=self.max_tokens,
            sampler=make_sampler(temp=0.1),
            verbose=False,
        )
        elapsed = time.time() - t0

        if trim_to_prefix:
            # Roll cache back to prefix only; offset attribute tracks current length.
            current = self.cache[0].offset
            n_trim = current - prefix_len
            if n_trim > 0:
                trim_prompt_cache(self.cache, n_trim)

        return parse_relation(raw), elapsed

    def teardown(self) -> None:
        self.model = None
        self.tokenizer = None
        self.cache = None
        self.prefix_tokens = []
        gc.collect()
        try:
            import mlx.core as mx
            mx.metal.clear_cache()
        except Exception:
            pass


def _gemma_batch_b16():
    return GemmaBatchBackend(name='gemma-batch-16', batch_size=16)


@dataclass
class GemmaBatchPrefixBackend:
    """Gemma combining batch_generate with a prefilled prefix KV cache.

    Setup is the same as GemmaPrefixCacheBackend (find shared prefix, prefill
    one cache). Per batch, the prefilled cache is deep-copied B times and passed
    to batch_generate alongside the per-sequence suffix tokens — so each
    sequence in the batch starts with the prefix already cached, and only the
    head_a/head_b suffix tokens go through prefill in parallel across the batch.
    """
    MODEL_ID: str = 'lmstudio-community/gemma-4-26B-A4B-it-MLX-4bit'
    name: str = 'gemma-batch-prefix'
    batch_size: int = 8
    max_tokens: int = 200
    model: Any = None
    tokenizer: Any = None
    prefix_tokens: list[int] = field(default_factory=list)
    prefix_cache: Any = None

    def setup(self) -> None:
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache
        import mlx.core as mx

        result = load(self.MODEL_ID)
        self.model, self.tokenizer = result[0], result[1]

        s1 = _gemma_format_prompt(self.tokenizer, 'alpha sample one', 'beta sample two')
        s2 = _gemma_format_prompt(self.tokenizer, 'gamma sample three', 'delta sample four')
        t1 = self.tokenizer.encode(s1)
        t2 = self.tokenizer.encode(s2)
        n = 0
        while n < min(len(t1), len(t2)) and t1[n] == t2[n]:
            n += 1
        self.prefix_tokens = t1[:n]
        print(
            f"  batch-prefix: shared prefix = {n} tokens, batch_size={self.batch_size}",
            flush=True,
        )

        self.prefix_cache = make_prompt_cache(self.model)
        self.model(mx.array(self.prefix_tokens)[None], cache=self.prefix_cache)
        mx.eval([c.state for c in self.prefix_cache])

    def predict_many(
        self, pairs: list[tuple[str, str, str]]
    ) -> list[tuple[str, float]]:
        from mlx_lm import batch_generate
        import copy as _copy

        prefix_len = len(self.prefix_tokens)

        # Tokenize all suffixes (or full prompts as fallback if prefix doesn't match).
        suffix_lists: list[list[int]] = []
        used_cache_flags: list[bool] = []
        for _label, a, b in pairs:
            full = _gemma_format_prompt(self.tokenizer, a, b)
            full_tokens = self.tokenizer.encode(full)
            if full_tokens[:prefix_len] == self.prefix_tokens:
                suffix_lists.append(full_tokens[prefix_len:])
                used_cache_flags.append(True)
            else:
                suffix_lists.append(full_tokens)
                used_cache_flags.append(False)

        results: list[tuple[str, float]] = []
        for i in range(0, len(suffix_lists), self.batch_size):
            chunk = suffix_lists[i:i + self.batch_size]
            chunk_flags = used_cache_flags[i:i + self.batch_size]

            # If every entry in chunk uses cache, pass per-sequence cache copies.
            # Mixed chunks fall back to no cache (rare; happens only on tokenizer drift).
            if all(chunk_flags):
                chunk_caches = [_copy.deepcopy(self.prefix_cache) for _ in range(len(chunk))]
            else:
                chunk_caches = None

            t0 = time.time()
            resp = batch_generate(
                self.model, self.tokenizer,
                prompts=chunk,
                prompt_caches=chunk_caches,
                max_tokens=self.max_tokens,
                verbose=False,
            )
            elapsed = time.time() - t0
            per_pair_lat = elapsed / len(chunk)
            for text in resp.texts:
                results.append((parse_relation(text), per_pair_lat))
            print(
                f"  batch[{i:>3}-{i + len(chunk):>3}] {len(chunk)} pairs in "
                f"{elapsed:5.2f}s ({per_pair_lat:.3f}s/pair, "
                f"{len(chunk) / elapsed:.2f} pairs/s)"
                + ("" if all(chunk_flags) else "  [no-cache fallback]"),
                flush=True,
            )
        return results

    def predict(self, a: str, b: str) -> tuple[str, float]:
        out = self.predict_many([("?", a, b)])
        return out[0]

    def teardown(self) -> None:
        self.model = None
        self.tokenizer = None
        self.prefix_cache = None
        self.prefix_tokens = []
        gc.collect()
        try:
            import mlx.core as mx
            mx.metal.clear_cache()
        except Exception:
            pass


MULTI_PAIR_PROMPT_HEADER = '''You are deciding the relation (if any) between pairs of memory heads.

For each pair, choose one of:
- "causes" — A causes B (or B is a consequence of A). Asymmetric.
- "contradicts" — A and B make incompatible claims. Symmetric.
- "related" — meaningfully connected but no specific relation. Symmetric.
- "none" — not actually related; the embedding similarity was misleading.

Output a JSON array with exactly one object per input pair, in the same order as the input. Each object has shape: {{"relation": "<one of the above>", "weight": <float 0-1>}}. Return ONLY the JSON array — no prose, no markdown fences.

Pairs:
'''


def _format_multi_pair_user_content(pairs_chunk: list[tuple[str, str, str]]) -> str:
    """Render N pairs into one multi-pair user message.
    Static instructions come first (cacheable prefix); pair list comes last."""
    lines = [MULTI_PAIR_PROMPT_HEADER]
    for idx, (_label, a, b) in enumerate(pairs_chunk, start=1):
        lines.append(f'{idx}. A: "{a}" B: "{b}"')
    return "\n".join(lines)


def _parse_multi_pair_array(raw: str, expected: int) -> list[str]:
    """Extract `expected` relation labels from a JSON array response.
    Falls back to scanning per-object regex if the array isn't clean JSON.
    Pads with PARSE_ERROR for any missing entries; truncates extras."""
    text = raw
    if '<channel|>' in text:
        text = text.split('<channel|>', 1)[1]
    text = re.sub(r'```(?:json)?\s*', '', text).replace('```', '')
    text = text.strip()
    # First try whole-string JSON.
    out: list[str] = []
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict) and 'relation' in item:
                    out.append(str(item['relation']))
    except json.JSONDecodeError:
        pass
    if len(out) != expected:
        # Fall back: scan for "relation":"<label>" occurrences in document order.
        out = re.findall(
            r'"relation"\s*:\s*"(causes|contradicts|related|none)"', text
        )
    # Pad / truncate to exactly `expected` entries.
    if len(out) < expected:
        out = out + ['PARSE_ERROR'] * (expected - len(out))
    elif len(out) > expected:
        out = out[:expected]
    return out


@dataclass
class GemmaMultiPairBackend:
    """Gemma combining batch + prefix-cache + multi-pair-per-prompt.

    Each LLM sequence in a batch carries `pairs_per_prompt` independent
    pairs and asks for one JSON array of N classifications. The shared
    instruction prefix is KV-cached across batch members, and `batch_size`
    sequences run in parallel via mlx_lm.batch_generate. So one
    batch_generate call processes `batch_size * pairs_per_prompt` pairs.

    Tradeoffs:
    - Pro: amortizes prefix prefill over more pairs; fewer total LLM calls.
    - Con: longer outputs per sequence (roughly 50 tokens × N), risk of
      degraded accuracy if the model loses track of pairs late in the list.
    """
    MODEL_ID: str = 'lmstudio-community/gemma-4-26B-A4B-it-MLX-4bit'
    name: str = 'gemma-multi-pair'
    batch_size: int = 4
    pairs_per_prompt: int = 10
    max_tokens: int = 1500
    model: Any = None
    tokenizer: Any = None
    prefix_tokens: list[int] = field(default_factory=list)
    prefix_cache: Any = None

    def _format_chat_for_pairs(self, pairs_chunk: list[tuple[str, str, str]]) -> str:
        text = _format_multi_pair_user_content(pairs_chunk)
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )

    def setup(self) -> None:
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache
        import mlx.core as mx

        result = load(self.MODEL_ID)
        self.model, self.tokenizer = result[0], result[1]

        # Build two distinct sample chunks of size pairs_per_prompt to find
        # the longest common token prefix (the static instructions block).
        sample_a = [
            ("?", f"alpha head a {i}", f"alpha head b {i}")
            for i in range(self.pairs_per_prompt)
        ]
        sample_b = [
            ("?", f"omega head a {i}", f"omega head b {i}")
            for i in range(self.pairs_per_prompt)
        ]
        t1 = self.tokenizer.encode(self._format_chat_for_pairs(sample_a))
        t2 = self.tokenizer.encode(self._format_chat_for_pairs(sample_b))
        n = 0
        while n < min(len(t1), len(t2)) and t1[n] == t2[n]:
            n += 1
        self.prefix_tokens = t1[:n]
        print(
            f"  multi-pair: pairs_per_prompt={self.pairs_per_prompt} "
            f"batch_size={self.batch_size} prefix={n} tokens "
            f"(of ~{len(t1)}-token sample prompt)",
            flush=True,
        )

        self.prefix_cache = make_prompt_cache(self.model)
        self.model(mx.array(self.prefix_tokens)[None], cache=self.prefix_cache)
        mx.eval([c.state for c in self.prefix_cache])

    def predict_many(
        self, pairs: list[tuple[str, str, str]]
    ) -> list[tuple[str, float]]:
        from mlx_lm import batch_generate
        import copy as _copy

        prefix_len = len(self.prefix_tokens)

        # Group pairs into chunks of pairs_per_prompt → each chunk becomes one
        # multi-pair prompt sequence.
        prompt_chunks: list[list[tuple[str, str, str]]] = []
        for i in range(0, len(pairs), self.pairs_per_prompt):
            prompt_chunks.append(pairs[i:i + self.pairs_per_prompt])

        # Tokenize each chunk's prompt; slice off shared prefix when matched.
        suffix_lists: list[list[int]] = []
        used_cache_flags: list[bool] = []
        for chunk in prompt_chunks:
            full_tokens = self.tokenizer.encode(self._format_chat_for_pairs(chunk))
            if full_tokens[:prefix_len] == self.prefix_tokens:
                suffix_lists.append(full_tokens[prefix_len:])
                used_cache_flags.append(True)
            else:
                suffix_lists.append(full_tokens)
                used_cache_flags.append(False)

        # Walk batch_size sequences at a time through batch_generate.
        results: list[tuple[str, float]] = []
        n_chunks = len(prompt_chunks)
        for i in range(0, n_chunks, self.batch_size):
            seq_slice = suffix_lists[i:i + self.batch_size]
            chunk_slice = prompt_chunks[i:i + self.batch_size]
            flags_slice = used_cache_flags[i:i + self.batch_size]
            if all(flags_slice):
                chunk_caches: list[list[Any]] | None = [
                    _copy.deepcopy(self.prefix_cache) for _ in seq_slice
                ]
            else:
                chunk_caches = None

            t0 = time.time()
            resp = batch_generate(
                self.model, self.tokenizer,
                prompts=seq_slice,
                prompt_caches=chunk_caches,
                max_tokens=self.max_tokens,
                verbose=False,
            )
            elapsed = time.time() - t0

            n_pairs_in_call = sum(len(c) for c in chunk_slice)
            per_pair_lat = elapsed / max(n_pairs_in_call, 1)
            for raw_text, sub_chunk in zip(resp.texts, chunk_slice, strict=True):
                labels = _parse_multi_pair_array(raw_text, expected=len(sub_chunk))
                for label in labels:
                    results.append((label, per_pair_lat))
            print(
                f"  multi-pair batch[{i:>3}-{i + len(seq_slice):>3}] "
                f"{len(seq_slice)} seqs × ~{self.pairs_per_prompt} pairs "
                f"({n_pairs_in_call} pairs total) in {elapsed:5.2f}s "
                f"({per_pair_lat:.3f}s/pair, {n_pairs_in_call / elapsed:.2f} pairs/s)"
                + ("" if all(flags_slice) else "  [no-cache fallback]"),
                flush=True,
            )
        return results

    def predict(self, a: str, b: str) -> tuple[str, float]:
        out = self.predict_many([("?", a, b)])
        return out[0]

    def teardown(self) -> None:
        self.model = None
        self.tokenizer = None
        self.prefix_cache = None
        self.prefix_tokens = []
        gc.collect()
        try:
            import mlx.core as mx
            mx.metal.clear_cache()
        except Exception:
            pass


def _gemma_multi_pair_5():
    return GemmaMultiPairBackend(name='gemma-multi-pair-5', pairs_per_prompt=5)


def _gemma_multi_pair_20():
    return GemmaMultiPairBackend(name='gemma-multi-pair-20', pairs_per_prompt=20, max_tokens=2500)


# NOTE: Gemma 4 dense variants (E2B / E4B) are intentionally absent here.
# As of mlx-lm 0.31.3 (and git main at the time of writing), the dense E2B
# and E4B checkpoints from lmstudio-community / mlx-community fail to load:
# the published weights carry per-layer k_proj/v_proj tensors on layers that
# mlx-lm's Gemma 4 model class treats as KV-shared, raising
# "Received N parameters not in model".
# Loading with strict=False bypasses the error but produces incoherent
# generations (the skipped weights are not actually redundant on the lmstudio
# ports). Re-add when an mlx-lm release lands proper Gemma 4 dense support.


@dataclass
class LMStudioBackend:
    """OpenAI-compatible client against a local LM Studio server.

    Lets us bench any model LM Studio can load (e.g. Gemma 4 dense E2B / E4B,
    which mlx-lm 0.31.x cannot load directly) without going through MLX from
    Python. The server runs the model in its own process; we just send chat
    completion requests over HTTP. No batch primitive — predict_many is a
    sequential loop.

    Start LM Studio, load a model, and enable "Local Server" (default port
    1234). Then run with:
        --backends lmstudio
        --backends lmstudio-gemma-e4b
        --backends lmstudio-gemma-e2b
    """
    name: str = 'lmstudio'
    base_url: str = 'http://localhost:1234/v1'
    model_id: str | None = None  # auto-detect first loaded model if None
    max_tokens: int = 200
    request_timeout_s: float = 120.0
    _session: Any = None

    def setup(self) -> None:
        import httpx
        self._session = httpx.Client(timeout=self.request_timeout_s)
        try:
            resp = self._session.get(f"{self.base_url}/models")
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(
                f"LM Studio server not reachable at {self.base_url}. "
                f"Start LM Studio, load a model, enable Local Server. ({e})"
            ) from e
        models = [m['id'] for m in resp.json().get('data', [])]
        if not models:
            raise RuntimeError(
                f"LM Studio at {self.base_url} returned no loaded models. "
                "Load a model in the UI first."
            )
        if self.model_id is None:
            self.model_id = models[0]
        elif self.model_id not in models:
            raise RuntimeError(
                f"LM Studio at {self.base_url} does not have {self.model_id!r} "
                f"loaded. Currently loaded: {models}"
            )
        print(f"  lmstudio: model={self.model_id}", flush=True)

    def predict(self, a: str, b: str) -> tuple[str, float]:
        prompt = PREFIX_OPTIMIZED_PROMPT.format(head_a=a, head_b=b)
        body = {
            'model': self.model_id,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.1,
            'max_tokens': self.max_tokens,
            'stream': False,
            # Disable Gemma 4's thinking trace at the chat-template level —
            # otherwise it consumes the entire token budget before emitting JSON
            # and we get finish_reason=length with empty content.
            'chat_template_kwargs': {'enable_thinking': False},
        }
        t0 = time.time()
        resp = self._session.post(f"{self.base_url}/chat/completions", json=body)
        elapsed = time.time() - t0
        if resp.status_code != 200:
            return 'EXCEPTION', elapsed
        try:
            content = resp.json()['choices'][0]['message']['content']
        except (KeyError, IndexError, ValueError):
            return 'PARSE_ERROR', elapsed
        return parse_relation(content), elapsed

    def teardown(self) -> None:
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
        self._session = None


def _lmstudio_gemma_e4b():
    return LMStudioBackend(
        name='lmstudio-gemma-e4b',
        model_id='gemma-4-e4b-it',  # LM Studio normalizes IDs to lowercase
    )


def _lmstudio_gemma_e2b():
    return LMStudioBackend(
        name='lmstudio-gemma-e2b',
        model_id='gemma-4-e2b-it',
    )


def _lmstudio_gemma_31b():
    return LMStudioBackend(
        name='lmstudio-gemma-31b',
        model_id='gemma-4-31b-it',
    )


@dataclass
class GeminiBackend:
    name: str = 'gemini-3-flash'
    llm: Any = None

    def setup(self) -> None:
        from hippo.config import load_api_key, load_config
        from hippo.models.llm import GeminiLLM
        cfg = load_config()
        key = load_api_key()
        if not key:
            raise RuntimeError(
                "No Gemini API key. Set GOOGLE_API_KEY or write to ~/.claude/hippo-secrets."
            )
        self.llm = GeminiLLM.load(
            api_key=key,
            model_id=cfg.gemini_model_id,
            default_thinking_level='minimal',
        )

    def predict(self, a: str, b: str) -> tuple[str, float]:
        prompt = SIMPLIFIED_PROMPT.format(head_a=a, head_b=b)
        t0 = time.time()
        raw = self.llm.generate_chat(
            [{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=200, thinking_level='minimal',
        )
        return parse_relation(raw), time.time() - t0

    def teardown(self) -> None:
        self.llm = None
        gc.collect()


# === Runner ===
@dataclass
class BackendResult:
    name: str
    setup_seconds: float
    predictions: list[tuple[str, str, str, float]] = field(default_factory=list)
    # tuples of (true_label, head_a_snippet, predicted, latency_s)

    @property
    def n(self) -> int: return len(self.predictions)
    @property
    def correct(self) -> int:
        return sum(1 for true, _, pred, _ in self.predictions if pred == true)
    @property
    def latencies(self) -> list[float]:
        return [lat for _, _, _, lat in self.predictions]

    def per_class_acc(self) -> dict[str, tuple[int, int]]:
        out: dict[str, list[tuple[int, int]]] = {}
        agg: dict[str, list[bool]] = {}
        for true, _, pred, _ in self.predictions:
            agg.setdefault(true, []).append(pred == true)
        return {k: (sum(v), len(v)) for k, v in agg.items()}


def run_backend(backend: Backend, fixtures: dict[str, list[tuple[str, str]]]) -> BackendResult:
    print(f"\n=== {backend.name} ===", flush=True)
    print("  setup...", flush=True)
    t0 = time.time()
    backend.setup()
    setup_s = time.time() - t0
    print(f"  setup done in {setup_s:.1f}s", flush=True)

    result = BackendResult(name=backend.name, setup_seconds=setup_s)

    # Flatten fixtures into ordered (true_label, a, b) list so batch and
    # per-call modes process the same sequence and slot results identically.
    flat: list[tuple[str, str, str]] = []
    for true_label, pairs in fixtures.items():
        for a, b in pairs:
            flat.append((true_label, a, b))

    run_t0 = time.time()
    if hasattr(backend, "predict_many"):
        try:
            outs = backend.predict_many(flat)  # type: ignore[attr-defined]
        except Exception as e:
            print(f"  ERROR in predict_many: {e}", flush=True)
            outs = [('EXCEPTION', 0.0)] * len(flat)
        for (true_label, a, _b), (pred, lat) in zip(flat, outs):
            result.predictions.append((true_label, a[:30], pred, lat))
            mark = '✓' if pred == true_label else '✗'
            print(f"  {mark} {true_label:<12} → {pred:<14} ({lat:5.3f}s)  {a[:48]}", flush=True)
    else:
        for true_label, a, b in flat:
            try:
                pred, lat = backend.predict(a, b)
            except Exception as e:
                print(f"  ERROR on pair ({a[:30]}...): {e}", flush=True)
                pred, lat = 'EXCEPTION', 0.0
            result.predictions.append((true_label, a[:30], pred, lat))
            mark = '✓' if pred == true_label else '✗'
            print(f"  {mark} {true_label:<12} → {pred:<14} ({lat:5.2f}s)  {a[:32]}|{b[:32]}", flush=True)
    run_wall = time.time() - run_t0

    print("  teardown...", flush=True)
    backend.teardown()
    print(f"  total wall-clock: {run_wall:.1f}s ({len(flat) / run_wall:.2f} pairs/s)", flush=True)
    return result


def report(results: list[BackendResult]) -> None:
    print("\n" + "=" * 88)
    print("SUMMARY")
    print("=" * 88)
    print(f"{'backend':<26} | {'acc':<11} | {'mean lat':>8} | {'p50':>7} | {'p95':>7} | {'setup':>6}")
    print('-' * 88)
    for r in results:
        if not r.predictions:
            continue
        lats = sorted(r.latencies)
        p50 = lats[len(lats) // 2]
        p95 = lats[int(len(lats) * 0.95)]
        mean = statistics.mean(lats)
        acc_pct = 100 * r.correct / r.n if r.n else 0
        print(f"{r.name:<26} | {r.correct}/{r.n} ({acc_pct:.0f}%) | {mean:7.2f}s | {p50:6.2f}s | {p95:6.2f}s | {r.setup_seconds:5.1f}s")

    classes = sorted({k for r in results for k in r.per_class_acc()})
    print("\nPer-class accuracy:")
    header = f"{'class':<14}" + ''.join(f" | {r.name:<26}" for r in results)
    print(header)
    for c in classes:
        row = f"{c:<14}"
        for r in results:
            stats = r.per_class_acc().get(c, (0, 0))
            row += f" | {stats[0]}/{stats[1]} ({100*stats[0]/stats[1] if stats[1] else 0:.0f}%)".ljust(29)
        print(row)

    print("\nDisagreements (where any backend disagrees with the label):")
    n = len(results[0].predictions) if results else 0
    for i in range(n):
        true = results[0].predictions[i][0]
        snippet = results[0].predictions[i][1]
        preds = [r.predictions[i][2] for r in results]
        if len(set(preds)) > 1 or any(p != true for p in preds):
            row = f"  label={true:<12}"
            for r, p in zip(results, preds):
                row += f"  {r.name.split('-')[0]}={p}"
            row += f"  | {snippet}..."
            print(row)


# === Persistence ===
def save_results(path: str, results: list[BackendResult]) -> None:
    blob = {
        'fixtures': FIXTURES,
        'results': [
            {
                'name': r.name,
                'setup_seconds': r.setup_seconds,
                'predictions': r.predictions,
            }
            for r in results
        ],
    }
    with open(path, 'w') as f:
        json.dump(blob, f, indent=2)
    print(f"\nSaved to {path}")


# === Main ===
BACKENDS: dict[str, Callable[[], Backend]] = {
    'qwen':           QwenBackend,
    'qwen3.5':        Qwen35Backend,
    'qwen3.5-think':  _qwen35_think,
    'gemma':          GemmaBackend,
    'gemma-think':    _gemma_think,
    'gemma-control':  GemmaControlBackend,
    'gemma-batch':    GemmaBatchBackend,
    'gemma-batch-16': _gemma_batch_b16,
    'gemma-prefix':   GemmaPrefixCacheBackend,
    'gemma-batch-prefix': GemmaBatchPrefixBackend,
    'gemma-multi-pair':   GemmaMultiPairBackend,
    'gemma-multi-pair-5': _gemma_multi_pair_5,
    'gemma-multi-pair-20': _gemma_multi_pair_20,
    'lmstudio':            LMStudioBackend,
    'lmstudio-gemma-e4b':  _lmstudio_gemma_e4b,
    'lmstudio-gemma-e2b':  _lmstudio_gemma_e2b,
    'lmstudio-gemma-31b':  _lmstudio_gemma_31b,
    'gemini':         GeminiBackend,
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--backends', default='gemini,qwen,gemma',
                   help='comma-separated subset of: ' + ','.join(BACKENDS.keys()))
    p.add_argument('--out', default='/tmp/bench_relation_classifiers.json')
    args = p.parse_args()

    selected = [b.strip() for b in args.backends.split(',') if b.strip()]
    for b in selected:
        if b not in BACKENDS:
            raise SystemExit(f"unknown backend: {b}. options: {list(BACKENDS)}")

    results: list[BackendResult] = []
    for b in selected:
        backend = BACKENDS[b]()
        try:
            results.append(run_backend(backend, FIXTURES))
        except Exception as e:
            print(f"Backend {b} failed entirely: {e}")

    report(results)
    save_results(args.out, results)


if __name__ == '__main__':
    main()
