Search.setIndex({"alltitles": {"An example cell": [[4, "an-example-cell"]], "Citations": [[3, "citations"]], "Code blocks and outputs": [[5, "code-blocks-and-outputs"]], "Content with notebooks": [[5, "content-with-notebooks"]], "Create a notebook with MyST Markdown": [[4, "create-a-notebook-with-myst-markdown"]], "Evaluation": [[0, "evaluation"]], "Finetuning large language models": [[0, "finetuning-large-language-models"]], "Finetuning the model": [[0, "finetuning-the-model"]], "Identify a dataset": [[0, "identify-a-dataset"]], "Installation": [[0, "installation"]], "Invoking the LLM for text classification": [[0, "invoking-the-llm-for-text-classification"]], "Learn more": [[3, "learn-more"]], "Markdown + notebooks": [[5, "markdown-notebooks"]], "Markdown Files": [[3, "markdown-files"]], "MyST markdown": [[5, "myst-markdown"]], "Notebooks with MyST Markdown": [[4, "notebooks-with-myst-markdown"]], "Prompt Engineering": [[0, "prompt-engineering"]], "Quickly add YAML metadata for MyST Notebooks": [[4, "quickly-add-yaml-metadata-for-myst-notebooks"]], "Rationale for different approaches": [[0, "rationale-for-different-approaches"]], "Reference": [[0, "reference"]], "Sample Roles and Directives": [[3, "sample-roles-and-directives"]], "Text Classification": [[0, "text-classification"]], "Welcome to the book on applying LLAMA 3": [[2, "welcome-to-the-book-on-applying-llama-3"]], "What is MyST?": [[3, "what-is-myst"]], "apply_llama3": [[1, "apply-llama3"]]}, "docnames": ["Finetuning-Large Language-models", "README", "intro", "markdown", "markdown-notebooks", "notebooks"], "envversion": {"sphinx": 61, "sphinx.domains.c": 3, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 9, "sphinx.domains.index": 1, "sphinx.domains.javascript": 3, "sphinx.domains.math": 2, "sphinx.domains.python": 4, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinxcontrib.bibtex": 9}, "filenames": ["Finetuning-Large Language-models.ipynb", "README.md", "intro.md", "markdown.md", "markdown-notebooks.md", "notebooks.ipynb"], "indexentries": {}, "objects": {}, "objnames": {}, "objtypes": {}, "terms": {"": [0, 2, 3, 4, 5], "0": [0, 5], "000": 0, "0x7f6f5d513b80": 5, "0x7fc83274ad40": 0, "1": [0, 2, 5], "10": [0, 5], "100": 5, "1003": 0, "1004": 0, "1005": 0, "1006": 0, "1007": 0, "1008": 0, "1009": 0, "1010": 0, "1011": 0, "1012": 0, "1014": 0, "1015": 0, "1016": 0, "1017": 0, "1019": 0, "1020": 0, "1022": 0, "1023": 0, "1024": 0, "1027": 0, "1028": 0, "1049": 0, "1050": 0, "1051": 0, "1053": 0, "1054": 0, "1055": 0, "1056": 0, "1057": 0, "1058": 0, "1059": 0, "120": 0, "120m": 0, "1226": 0, "1227": 0, "1228": 0, "1235": 0, "1236": 0, "1237": 0, "1238": 0, "1239": 0, "1240": 0, "159": 0, "160": 0, "161": 0, "165": 0, "166": 0, "167": 0, "168": 0, "169": 0, "17": 0, "170": 0, "171": 0, "172": 0, "173": 0, "174": 0, "175": 0, "176": 0, "177": 0, "178": 0, "179": 0, "180": 0, "19680801": 5, "2": [0, 2, 4], "2014": 3, "2023": 0, "2493": 0, "2494": 0, "2495": 0, "2496": 0, "2497": 0, "275": 0, "276": 0, "277": 0, "3": 0, "36": 0, "4": [0, 4, 5], "429": 0, "443": 0, "444": 0, "445": 0, "446": 0, "447": 0, "448": 0, "449": 0, "450": 0, "451": 0, "452": 0, "453": 0, "454": 0, "455": 0, "4552": 0, "4553": 0, "4554": 0, "4555": 0, "4556": 0, "4557": 0, "4558": 0, "4559": 0, "456": 0, "4560": 0, "4561": 0, "4562": 0, "457": 0, "458": 0, "459": 0, "460": 0, "461": 0, "5": 5, "500": 0, "53": 0, "535": 0, "536": 0, "537": 0, "538": 0, "573": 0, "574": 0, "575": 0, "591": 0, "592": 0, "593": 0, "596": 0, "597": 0, "598": 0, "599": 0, "603": 0, "604": 0, "605": 0, "606": 0, "607": 0, "608": 0, "609": 0, "610": 0, "611": 0, "612": 0, "613": 0, "614": 0, "615": 0, "616": 0, "617": 0, "618": 0, "619": 0, "620": 0, "621": 0, "622": 0, "623": 0, "624": 0, "625": 0, "626": 0, "627": 0, "628": 0, "629": 0, "630": 0, "631": 0, "632": 0, "633": 0, "634": 0, "635": 0, "636": 0, "637": 0, "638": 0, "639": 0, "640": 0, "641": 0, "669": 0, "670": 0, "671": 0, "672": 0, "673": 0, "674": 0, "675": 0, "7": 0, "8": 0, "912": 0, "913": 0, "914": 0, "919": 0, "920": 0, "921": 0, "922": 0, "923": 0, "924": 0, "925": 0, "926": 0, "927": 0, "A": 0, "As": [0, 5], "But": [0, 5], "By": 0, "For": [3, 5], "If": [0, 2, 4], "In": [0, 3], "Into": 0, "It": [0, 2, 3], "No": [], "OR": 0, "On": 0, "That": 4, "The": [0, 3, 4], "There": [0, 5], "With": 4, "_base_cli": 0, "_combine_llm_output": 0, "_convert_input": 0, "_create_chat_result": 0, "_create_message_dict": 0, "_gener": 0, "_generate_with_cach": 0, "_make_status_error_from_respons": 0, "_map_singl": 0, "_merge_config": 0, "_post": 0, "_process_respons": 0, "_request": 0, "_retry_request": 0, "_should_retri": 0, "_streamt": 0, "_util": 0, "abbrev": 0, "abil": 0, "about": [0, 2, 3, 4, 5], "abov": 0, "acceler": 0, "accept": 3, "access": 0, "accord": 0, "account": 0, "accuraci": 0, "accuracy_metr": 0, "achiev": 0, "aclanthologi": 0, "actual": 0, "adapt": 0, "add_": 5, "addition": 0, "advantag": 0, "ag": 0, "ag_new": 0, "again": 0, "align": 5, "all": [0, 3, 4], "allow": [0, 3], "alreadi": 2, "also": [0, 2, 3, 4, 5], "altern": 0, "alwai": 0, "amount": 0, "an": [0, 2, 3], "ani": [0, 2, 4], "announc": 0, "anoth": 0, "ansari": 0, "append": 0, "applic": 0, "ar": [0, 2, 3, 4], "area": 0, "arg": 0, "argument": 0, "arrai": [0, 5], "arrow_dataset": 0, "articl": 0, "assist": 0, "attempt": 0, "auditori": 3, "augment": 0, "australia": 3, "avail": 0, "ax": 5, "back": 0, "band": 0, "base": [0, 4], "basechatmodel": 0, "basechatopenai": 0, "baseexcept": 0, "baselin": 0, "basemessag": 0, "basemodel": 0, "bear": 0, "befor": 0, "begin": 5, "being": 3, "below": 0, "benchmark": 0, "benefit": 0, "bert": 0, "better": 0, "bib": 3, "bibliographi": 3, "bibtex": 3, "bitsandbyt": 0, "black": 0, "block": [0, 4], "bloomberggpt": 0, "bodi": 0, "book": [1, 3, 4, 5], "both": 3, "bound": 0, "box": [0, 3], "brian": 3, "brisban": 3, "build": [0, 3], "built": 4, "bundl": [], "busi": [0, 2], "cach": 0, "call": [0, 3], "callback": 0, "can": [0, 2, 3, 4, 5], "canada": 0, "capabl": 0, "captur": 0, "case": 0, "cast": 0, "cast_to": 0, "categori": 0, "cell": 0, "chapter": [0, 2], "chat": 0, "chat_model": 0, "chatcomplet": 0, "chatcompletionchunk": 0, "chatgener": 0, "chatgpt": 0, "chatopenai": 0, "chatprompttempl": 0, "check": 5, "christoph": 3, "cite": 3, "class": 0, "class1": 0, "classifi": 0, "claw": 0, "client": 0, "close": 0, "cloudspac": 0, "cm": 5, "cmap": 5, "code": [0, 3, 4], "cognit": 3, "cold": 5, "color": 5, "com": 0, "combin": 0, "command": 4, "commerci": 0, "commonmark": 3, "commun": 0, "compar": 0, "compel": 0, "compet": 0, "complet": 0, "completion_create_param": 0, "completioncreateparam": 0, "comprehens": 0, "comput": 0, "confer": 3, "config": 0, "consequ": 0, "consid": 0, "construct": 0, "contact": 2, "contain": 0, "content": [2, 3, 4], "contest": 0, "context": 0, "contextlib": 5, "convers": 0, "convert": 4, "coolwarm": 5, "corpu": 0, "correctli": 0, "correspond": 0, "cortex": 3, "cost": 0, "couldn": 0, "courtesi": 0, "cover": 2, "creat": [0, 5], "crucial": 0, "custom_lin": 5, "cycler": 5, "cynic": 0, "d": 4, "dai": 0, "data": [0, 2, 5], "dataset_process": 0, "date": 0, "de": 3, "debug": 0, "decad": 0, "decod": 0, "def": 0, "default": [0, 4], "defin": [0, 4], "delv": 0, "demonstr": 0, "depend": 3, "depth": [], "detail": 4, "determinist": 0, "dictionari": 0, "differ": 3, "dill": 0, "direct": 4, "discuss": 0, "displai": 4, "do": [0, 3, 5], "document": [0, 3, 4, 5], "doe": [], "dollar": 5, "domain": 0, "dotenv": 0, "downstream": 0, "dure": 0, "dwindl": 0, "e": [0, 2], "each": 0, "earlier": 0, "easili": 0, "ecosystem": 3, "effect": 0, "effici": 0, "els": 0, "emb": 5, "emerg": 0, "emnlp": 0, "emploi": 0, "emul": 0, "enabl": 0, "encod": 0, "end": 5, "engag": 0, "enhanc": 0, "enough": 0, "ensure_config": 0, "entir": 0, "entiti": 0, "enum": 0, "enumer": 0, "enumoutputpars": 0, "env": 0, "era": 0, "err": 0, "error": 0, "escap": 5, "establish": 0, "etc": 5, "everybodi": 0, "everydai": 0, "everyth": 0, "evid": 3, "exampl": [0, 3, 5], "excel": 0, "except": 0, "execut": 4, "exhibit": 0, "exitstack": 5, "expand": 0, "explicit": 0, "explicitli": 0, "explor": 0, "extens": 3, "extra_bodi": 0, "extra_head": 0, "extra_queri": 0, "extract": 0, "f": 0, "failur": 0, "fals": 0, "familiar": 2, "featur": 0, "feedback": 2, "feel": [], "few": 0, "field": 0, "fig": 5, "figsiz": 5, "file": [0, 2, 4], "final": 0, "finalrequestopt": 0, "financi": 0, "find": 0, "fine": 0, "finetun": 2, "fingerprint": 0, "finish": 0, "first": 0, "fix": 5, "flattened_output": 0, "flavor": 3, "flight": 0, "float": 0, "flow": 0, "follow": [0, 2, 3, 4], "format": 0, "foundat": 0, "four": 0, "framework": [0, 2], "frequency_penalti": 0, "from": [0, 2, 5], "from_templ": 0, "frontier": 3, "full": 0, "func": 0, "function": [0, 3], "function_cal": 0, "fund": 0, "further": 0, "gain": 0, "gener": 0, "generate_prompt": 0, "get": [0, 3, 4], "github": 1, "give": 0, "given": 0, "go": [], "good": 0, "gpt": 0, "gpu": 0, "great": 0, "green": 0, "guid": [0, 5], "ha": 0, "handl": 2, "hash": 0, "have": [2, 4], "hdhpk14": 3, "header": 0, "healthcar": 0, "heer": 3, "help": 3, "here": [0, 1, 2, 3, 5], "hf": 0, "highli": 0, "holdgraf": 3, "holdgraf_evidence_2014": 3, "home": 0, "hot": 5, "how": [0, 2, 4], "howev": 0, "html": [1, 5], "http": [0, 1], "httpx": 0, "human": [0, 3], "i": [0, 1, 2, 4, 5], "id_to_label": 0, "ignor": 0, "ii": 5, "imag": 5, "import": [0, 5], "improv": [0, 2], "includ": [4, 5], "incorpor": 0, "industri": 0, "infer": 0, "inform": [0, 4, 5], "inherit": 0, "init": 4, "initi": 0, "inlin": 3, "inner": 0, "inp": 0, "input": [0, 3], "insert": 3, "insight": 0, "inspect": 0, "instal": 2, "instead": 0, "instruct": [0, 4], "integ": 0, "intend": 2, "interact": [0, 5], "interfac": 0, "intern": 3, "intro": 1, "introduct": 2, "invalid": 0, "involv": 0, "io": 1, "ion": 5, "iphon": 0, "ipynb": 3, "is_clos": 0, "item": 0, "its": 0, "json_data": 0, "jupyt": [3, 4, 5], "jupyterbook": 3, "jupytext": 4, "just": [0, 3], "k2pw68cqf90u1datxdpox4ft": 0, "keep": 5, "kei": 0, "kernel": 4, "kind": 3, "knight": 3, "know": 0, "knowledg": 0, "known": 0, "kwarg": 0, "l": 0, "la_": 5, "label": 0, "langchain": 0, "langchain_cor": 0, "langchain_openai": 0, "languag": [2, 3], "language_model": 0, "languagemodelinput": 0, "larg": 2, "last": 0, "launch": 0, "lead": 0, "learn": [0, 2], "legend": 5, "less": 0, "let": [0, 4], "leverag": 0, "lib": 0, "librari": 0, "like": [0, 3, 4], "limit": 0, "line": [0, 3, 4, 5], "line2d": 5, "linspac": 5, "list": 0, "literatur": 0, "littl": 0, "ll": [0, 3], "llama": 0, "llm_output": 0, "llmresult": 0, "load": 0, "load_dataset": 0, "load_dotenv": 0, "local": 0, "log": 0, "logit_bia": 0, "logprob": 0, "logspac": 5, "look": 0, "lot": [3, 5], "lower": 0, "lowercas": 0, "lw": 5, "m": 0, "mai": 0, "main": 0, "major": [], "make": [0, 5], "make_request_opt": 0, "man": 0, "mani": [3, 4], "map": 0, "markdown": 2, "markdownfil": 4, "markedli": 3, "markup": 3, "mask": 0, "massiv": 0, "math": 5, "matplotlib": 5, "max_token": 0, "maybe_transform": 0, "mbox": 5, "mc": 2, "md": [3, 4], "mean": [0, 5], "mechan": 0, "medic": 0, "medium": 5, "mention": 0, "messag": 0, "message_dict": 0, "metadata": 0, "method": 0, "metric": 0, "might": 0, "million": 0, "min": 0, "mine": 0, "miniconda3": 0, "miss": 0, "model": 2, "modifi": 0, "modul": [], "modulenotfounderror": [], "more": [0, 2, 4, 5], "moreov": 3, "most": [0, 3], "msg": 0, "multi": 0, "multipl": 0, "must": 3, "myst": 2, "n": [0, 3, 5], "name": 0, "natur": 0, "necessari": 0, "need": [0, 4], "neurosci": 3, "new": [0, 2], "next": 0, "nlp": 0, "non": 0, "none": 0, "not_given": 0, "note": [0, 2, 3], "notebook": [2, 3], "notgiven": 0, "np": 5, "nuanc": 0, "num_proc": 0, "numpi": 5, "o": 0, "object": 0, "obtain": 0, "off": [3, 4], "offici": 0, "on_llm_error": 0, "onc": 0, "one": [0, 3], "onli": 0, "openai": 0, "opt": 0, "optim": 0, "option": 0, "org": [0, 3], "organ": [0, 2], "other": [0, 4], "our": 0, "out": [0, 5], "output": [0, 4], "output_pars": 0, "overview": 3, "own": 2, "p": 0, "packag": 0, "page": [3, 4], "parallel_tool_cal": 0, "param": 0, "paramet": 0, "particular": 0, "paslei": 3, "passag": 0, "path": [0, 4], "pdf": 0, "per": 0, "perform": 0, "pick": 0, "pickl": 0, "pip": 0, "pipelin": 0, "platform": 0, "pleas": [0, 2], "plot": 5, "plt": 5, "pmaxit": 1, "pop": 0, "post": [0, 5], "potenti": 0, "power": 3, "practic": 2, "pre": 0, "predict": [0, 3], "presenc": 4, "presence_penalti": 0, "present": 0, "pretrain": 0, "preview": 0, "previou": 0, "primari": 0, "primarili": 0, "print": [0, 4], "privat": 0, "prize": 0, "process": 0, "process_text": 0, "profession": 2, "progress": 2, "prompt_messag": 0, "promptvalu": 0, "prop_cycl": 5, "properli": [0, 3], "properti": 0, "provid": 0, "publicli": 0, "purpos": [0, 3], "py": 0, "pydantic_v1": 0, "pyplot": 5, "python3": 0, "question": [0, 2], "quot": 0, "r": 0, "race": 0, "rag": 0, "rais": 0, "ramsai": 3, "randn": 5, "random": [0, 5], "rang": [0, 5], "rate": 0, "rate_limit_exceed": 0, "ratelimiterror": 0, "rather": 0, "raw_dataset": 0, "rcparam": 5, "re": 0, "reach": 0, "read": 0, "reader": 2, "recent": 0, "recognit": 0, "recomput": 0, "reduc": 0, "refer": 3, "refin": 0, "regular": 3, "relat": 0, "relev": 0, "remain": 0, "remaining_retri": 0, "remark": 0, "render": 3, "repeat": 0, "repres": 0, "represent": 0, "reproduc": 5, "request": 0, "requir": 0, "required_arg": 0, "research": [0, 2], "resourc": 0, "respons": 0, "response_format": 0, "response_head": 0, "responset": 0, "rest": 4, "result": 0, "retri": 0, "retriev": 0, "return": 0, "reus": 0, "reuter": 0, "rnn": 0, "robert": 3, "rocket": 0, "root": 0, "rpm": 0, "run": [0, 4], "run_id": 0, "run_manag": 0, "run_nam": 0, "runnabl": 0, "runnablebindingbas": 0, "runnableconfig": 0, "runnablesequ": 0, "same": [0, 3], "sampl": [0, 5], "save": 0, "save_to_disk": 0, "scheme": 0, "sci": 0, "scienc": 0, "scratch": 0, "seamlessli": 0, "second": 0, "see": [0, 3, 4, 5], "seed": [0, 5], "self": 0, "seller": 0, "sentenc": 0, "sequenc": 0, "seri": 0, "serializ": 0, "serv": 3, "set": 0, "setup": 0, "share": 0, "short": 0, "shot": 0, "should": [0, 4], "show": [0, 3, 4], "sign": 5, "signatur": 0, "signific": 0, "significantli": 0, "similar": 3, "simpl": 3, "simultan": 0, "sinc": 0, "singl": 0, "site": 0, "sleep": 0, "slight": 3, "small": 3, "smaller": 0, "so": 4, "sole": 0, "solid": 0, "solv": 0, "some": [0, 3, 5], "space": 0, "spaceflight": 0, "span": 3, "special": [0, 3], "specif": [0, 3], "specifi": 0, "sphinx": 3, "sport": 0, "st": 0, "stai": 0, "stand": 3, "start": [0, 3, 4], "starter": 3, "state": 5, "statu": 0, "step": 0, "stop": 0, "store": [0, 3], "str": 0, "stream": 0, "stream_cl": 0, "stream_opt": 0, "street": 0, "strength": 0, "structur": [0, 3], "struggl": 0, "style": 0, "suborbit": 0, "subplot": 5, "subsequ": 0, "suggest": 2, "supervis": 0, "support": 4, "sure": [0, 5], "syncapicli": 0, "synchron": 0, "synergi": 0, "syntax": [0, 3], "syntaxerror": 0, "system": 0, "t": [0, 3, 5], "tag": 0, "tagging_chain": 0, "tagging_prompt": 0, "tailor": 0, "take": 0, "task": 0, "team": 0, "tech": 0, "techiqu": 0, "techniqu": 0, "technologi": 0, "temperatur": 0, "test": 0, "tex": 5, "text": [3, 4], "than": 0, "thei": [0, 3], "thi": [0, 2, 3, 4, 5], "thing": 4, "those": [2, 3], "thousand": 0, "thread": 0, "through": 0, "tht": 0, "time": 0, "timeout": 0, "to_httpx_fil": 0, "to_messag": 0, "token": 0, "tool": [0, 3], "tool_choic": 0, "top": 4, "top_logprob": 0, "top_p": 0, "topic": [0, 2], "toronto": 0, "traceback": 0, "train": 0, "transfer": 0, "transform": 0, "translat": 0, "treat": 4, "true": 0, "try": 0, "tune": 0, "turbo": 0, "two": [0, 3, 4], "type": 0, "typeerror": 0, "u": [0, 2], "ultra": 0, "understand": [0, 4], "unleash": 0, "unsupervis": 0, "up": 0, "url": 0, "us": [0, 2, 3, 4], "usecas": 2, "user": 0, "variat": 3, "variou": [0, 2], "vast": 0, "veri": 0, "versatil": 0, "via": 0, "viabl": 0, "vibrant": 0, "visit": 0, "vocabulari": 0, "wa": 0, "wai": 0, "wall": 0, "want": [0, 2, 5], "warn": 0, "we": [0, 2], "web": 0, "well": [0, 5], "wendi": 3, "when": [0, 3, 4], "where": 0, "wherea": 3, "whether": 3, "which": [0, 4], "while": 0, "who": 2, "wide": 0, "with_structured_output": 0, "without": 0, "won": 0, "word": 0, "work": [0, 2, 5], "world": 0, "wrapper": 0, "write": [3, 4], "written": [3, 4], "x": 0, "you": [0, 2, 3, 4, 5], "your": [0, 2, 3, 4, 5], "zero": 0, "zeu": 0}, "titles": ["Installation", "apply_llama3", "Welcome to the book on applying LLAMA 3", "Markdown Files", "Notebooks with MyST Markdown", "Content with notebooks"], "titleterms": {"3": 2, "add": 4, "an": 4, "appli": 2, "apply_llama3": 1, "approach": 0, "block": 5, "book": 2, "cell": 4, "citat": 3, "classif": 0, "code": 5, "content": 5, "creat": 4, "dataset": 0, "differ": 0, "direct": 3, "engin": 0, "evalu": 0, "exampl": 4, "file": 3, "finetun": 0, "i": 3, "identifi": 0, "instal": 0, "invok": 0, "jupyt": [], "languag": 0, "larg": 0, "learn": 3, "llama": 2, "llm": 0, "markdown": [3, 4, 5], "metadata": 4, "model": 0, "more": 3, "myst": [3, 4, 5], "notebook": [4, 5], "output": 5, "prompt": 0, "quickli": 4, "rational": 0, "refer": 0, "role": 3, "sampl": 3, "text": 0, "welcom": 2, "what": 3, "yaml": 4, "your": []}})