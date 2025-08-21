# LLM-ThreatLab
LLMThreatLab is a Python Streamlit app for LLM security scanning. It automates red-teaming by running predefined attacks, analyzing model responses for vulnerabilities, and generating reports. It features a modular design with a unified interface for various LLMs, an attack/detection framework, and an interactive UI.
________________________________________
LLMThreatLab: Technical Codebase Analysis
Document Version: 1.0
Date: August 21, 2025
________________________________________
1.0 Executive Summary
sLLMThreatLab, a Python-based web application built with Streamlit. The application serves as a comprehensive security and vulnerability scanner for Large Language Models (LLMs). Its primary function is to automate "red-teaming" activities by subjecting LLMs to a battery of predefined security attacks, analyzing the responses for vulnerabilities, and presenting the findings in a clear, actionable report.
The architecture is designed for modularity and extensibility, featuring three core components:
1.	Model Abstraction Layer: A unified interface for interacting with various LLMs from different providers (OpenAI, Hugging Face, local files).
2.	Attack & Detection Framework: A systematic collection of functions that craft malicious prompts (attacks) and corresponding heuristic functions that analyze model outputs for signs of compromise (detections).
3.	Interactive User Interface (UI): A Streamlit-based web interface that allows users to configure models, select attacks, run tests, and review detailed results.
This document will walk through each component, explaining its key functions, classes, and the overall execution flow of the application.
________________________________________
2.0 Core Architectural Components
The application is logically divided into several key areas that work together to provide its functionality.
The script begins by importing necessary libraries and defining global constants.
•	Core Libraries:
o	streamlit: The foundational framework for building the interactive web UI.
o	asyncio & nest_asyncio: Essential for managing asynchronous operations, enabling non-blocking API calls to services like OpenAI, which prevents the UI from freezing.
o	transformers & huggingface_hub: The Hugging Face ecosystem libraries used to download, cache, and run open-source models.
o	openai: The official library for interacting with OpenAI's suite of models (GPT-3.5, GPT-4).
o	pandas: Used for structuring summary results into a clean, tabular format.
•	Configuration:
o	MODEL_MAP: A dictionary mapping user-friendly model names to their technical identifiers and providers.
o	Default Parameters: Constants like DEFAULT_MAX_TOKENS and DEFAULT_TEMPERATURE provide baseline settings for model generation.
o	ThreadPoolExecutor: Manages a pool of threads to run synchronous Hugging Face models without blocking the main application thread.
A critical design feature is the abstraction of model interactions into a single, consistent interface. This is primarily achieved through the ModelWrapper class.
•	@st.cache_resource: This Streamlit decorator is applied to model-loading functions (get_openai_client, load_hf_pipeline). It caches the loaded model or API client in memory, ensuring that large, time-consuming models are loaded only once per session.
•	ModelWrapper Class:
o	__init__(...): The constructor initializes the wrapper based on the specified provider ('openai' or 'hf'). It handles the logic for creating an OpenAI client or loading a Hugging Face pipeline.
o	async generate(...): This method is the public-facing interface for text generation. It accepts a prompt and generation parameters. Internally, it handles the provider-specific logic:
	For OpenAI, it makes an asynchronous API call.
	For Hugging Face, it runs the synchronous pipeline in a separate thread using the ThreadPoolExecutor to avoid blocking the event loop.
This design allows the rest of the application to generate text from any model using a single, simple function call.
This component is responsible for crafting the specialized prompts used to test for vulnerabilities.
•	Attack Functions (attack_*): Each function (e.g., attack_jailbreak, attack_bias) is designed to probe a specific vulnerability. It takes a benign base prompt as input and modifies it to create a malicious or probing prompt.
•	Return Value: Every attack function returns a tuple containing:
1.	The full, malicious attack prompt (a string).
2.	A metadata dictionary (meta) containing the attack type, markers to look for in the response, and other relevant context.
•	Sweep Attacks: Attacks like "SSTI Sweep" and "Prompt Injection Sweep" are more advanced. They are handled by dedicated builder functions (build_ssti_attack_variants, build_prompt_injection_variants) that generate a list of many different prompt variations to test a wider range of potential exploits.
For each attack, there is a corresponding detection function that analyzes the model's output.
•	Detector Functions (detect_*): Each function (e.g., detect_jailbreak_vulnerability, detect_ssti_vulnerability) takes the model's generated text and the attack's metadata as input.
•	Core Logic: The detectors employ a range of techniques, including:
o	Keyword Matching: Searching for specific words or phrases (e.g., toxic language, confidential markers).
o	Regular Expressions: Using complex patterns to find evidence of compromise (e.g., the "DAN" jailbreak confirmation).
o	Heuristic Analysis: Applying rules of thumb to identify suspicious behavior (e.g., if the model claims it executed code).
•	Content Isolation: A key feature is the logic to intelligently strip the original prompt from the model's output before analysis. This prevents the detector from being confused by the contents of the prompt itself, focusing only on the newly generated text.
•	Universal Harmful Content Scanner (detect_harmful_content): This special detector acts as a safety net. It runs on every generated response, regardless of the attack type, to scan for a broad range of harmful content, including illegal instructions, violence, and severe toxicity.
________________________________________
3.0 Execution Flow and User Experience
The application's logic is driven by user interaction through the Streamlit UI.
1.	Configuration (Sidebar): The user configures the test session via the sidebar. This includes:
o	Selecting a target model and providing necessary credentials (API keys, tokens).
o	Adjusting text generation parameters.
o	Choosing one or more attacks to run.
o	Providing a base prompt (or a batch of prompts via CSV).
o	Confirming ethical use of the tool.
2.	Initiating the Run: The user clicks the "Run selected attacks" button.
3.	Orchestration (run_attacks_on_model): This triggers the main orchestration function, which executes the following loop:
o	It iterates through each base prompt.
o	For each prompt, it iterates through each selected attack.
o	It calls the appropriate attack_* or build_*_variants function to create the malicious prompt(s).
o	It calls the model_wrapper.generate() method to get the model's response.
o	It passes the response to the relevant detect_* functions for analysis.
o	It compiles a comprehensive record of the entire transaction (inputs, outputs, findings, metadata).
4.	Displaying Results: Once the run is complete, the results are presented to the user in the main area of the application:
o	Summary Table: A pandas DataFrame provides a high-level, sortable overview of all tests and their outcomes (e.g., jailbroken: True).
o	At-a-Glance Summary: A series of colored status boxes (st.error, st.success) give an immediate pass/fail summary for each attack category.
o	Detailed Expanders: Each individual test run can be inspected in detail, showing the exact prompts, the full model response, and the specific reasons for the vulnerability findings.
o	Data Export: The user can download the summary as a CSV or the complete, raw results as a JSON file for further analysis.
________________________________________
4.0 Conclusion
LLMThreatLab is a powerful and well-architected tool for the defensive security research of Large Language Models. Its modular design, abstraction of model backends, and comprehensive attack-and-detect framework make it highly effective. Key innovations like the universal harmful content scanner and the intelligent isolation of model-generated content enhance the accuracy and reliability of its findings. The user-friendly Streamlit interface makes this sophisticated testing accessible to a broad audience of developers, researchers, and security professionals.

