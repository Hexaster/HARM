"""Prompt templates from the paper (Section 3.4), verbatim.

Placeholders use Python str.format() syntax (single braces) in place of the
paper's {{double-brace}} notation.
"""

GENERATE_RAW_CLINICAL_NOTE_PROMPT = """You are an experienced medical expert skilled in drafting standardized medical course records based on diseases and key consultation points. Please use the provided disease information and corresponding consultation points, along with the given template and supplied knowledge, to compose a standardized medical course record for this disease.

Below is the knowledge to this disease:
{disease}
{diagnostic_key_points}

Below is the template for the clinical note:
Medical history:
Physical examination:
Auxiliary examination:
Case characteristics:
Initial diagnosis:
Diagnostic basis:
Differential diagnosis process:
Final diagnosis:"""

PATIENT_INFORMATION_EXTRACTION_PROMPT = """You are an experienced clinical note specialist, adept at extracting the medical history, physical examination, and auxiliary examination information from data provided by patient. Please use the information provided by the patient to systematically consider and itemize the medical history, physical examination, and auxiliary examinations. If certain data are not provided, mark the corresponding section as 'None' without making additional assumptions.

Below is the patient's question:
{question}"""

ANALYSIS_AND_SUMMARIZE_PROMPT = """You are an experienced medical analysis expert, skilled in comprehensively analyzing, summarizing, and organizing a patient's medical history, physical examination, and auxiliary examination to document the patient's clinical features. Please carefully review the patient's issues and itemize the clinical features, including positive findings and negative symptoms and signs relevant for differential diagnosis. Be sure to use only the provided information, without referencing external data.

Below is the medical history, physical examination, and auxiliary examination to this patient:
{medical_history}
{physical_examination}
{auxiliary_examination}

Below is the patient's question:
{question}"""

MAKE_PRELIMINARY_DIAGNOSIS_PROMPT = """You are an experienced clinical diagnosis expert, skilled in making preliminary diagnoses and analyses based on provided patient clinical features. Please provide a preliminary diagnosis based on the patient's case features and detail the diagnostic basis point by point.

Below is the clinical features to this patient:
{clinical_features}

Below is the patient's question:
{question}"""

REFLECT_PRELIMINARY_DIAGNOSIS_PROMPT = """You are an experienced clinical review expert, skilled in evaluating the diagnostic validity of clinical notes based on key inquiry points for diseases. Please thoroughly review the key inquiry points of the preliminary diagnosis provided and assess whether the preliminary diagnosis and diagnostic basis in the clinical note align with these points.

If deemed unreasonable, output the result as a JSON-formatted Dict{{"flag": false, "diagnosis_error": Str(Reasons for diagnostic errors)}}.

Below is the preliminary diagnosis and diagnostic basis:
{preliminary_diagnosis}
{diagnostic_basis}

Below is the key inquiry points:
{key_inquiry_points}"""

DIFFERENTIAL_DIAGNOSIS_PROMPT = """You are an experienced differential diagnosis expert, skilled in systematically analyzing key inquiry points to rule out diseases. Please carefully review the inquiry points of the diseases requiring differentiation and conduct a step-by-step differential diagnosis based on the patient's clinical note.

Document the differential diagnosis process point by point and output it in JSON format as Dict{{"diff_process": Str(differential diagnosis process)}}.

Below is the list of diseases to be ruled out through differential diagnosis:
{diseases_list}

Below is the key inquiry points to these diseases:
{key_inquiry_points}"""

REFLECT_DIFFERENTIAL_DIAGNOSIS_PROCESS_PROMPT = """You are an experienced clinical differential diagnosis expert, skilled in reflecting on and evaluating the rationality of differential diagnosis processes. Please reflect on the differential diagnosis process and assess whether the differentiation for each disease is reasonable.

If it is reasonable, output in JSON format as Dict{{"flag":true, "Final_Diagnosis": Str(final diagnosis)}}.
Otherwise, output in JSON format as Dict{{"flag":false, "diff_error": Str(Diseases requiring rediagnosis)}}.

Below is the list of diseases to be ruled out through differential diagnosis, along with the corresponding diagnostic process.
{diseases_list}
{differential_diagnosis_process}"""

REFLECT_AND_CORRECT_ICA_PROMPT = """You are an experienced expert in reviewing clinical notes, skilled in comparing raw clinical note with a given standardized template. Now, please compare the obtained raw clinical note with the given standardized clinical note template. The part that needs to be analyzed is the medical history, physical examination, auxiliary examination, and clinical features. If you find any part to be unreasonable, provide suggestions for improvement, and output in JSON format as Dict{{"flag":false, "ICA_error": Str(suggestions for improvement)}}.

Below is the raw clinical note.
{raw_clinical_note}

Below is a standardized template for a standardized clinical note of the final diagnosis.
{standardized_clinical_note}"""

REFLECT_AND_CORRECT_PDA_PROMPT = """You are an experienced expert in reviewing clinical notes, skilled in comparing raw clinical note with a given standardized template. Now, please compare the obtained raw clinical note with the given standardized clinical note template. The part that needs to be analyzed is the preliminary diagnosis and diagnostic basis. If you think this part is unreasonable, please give suggestions for improvement. , and output in JSON format as Dict{{"flag":false, "PDA_error": Str(suggestions for improvement)}}.

Below is the raw clinical note.
{raw_clinical_note}

Below is a standardized template for a standardized clinical note of the final diagnosis.
{standardized_clinical_note}"""

REFLECT_AND_CORRECT_DDA_PROMPT = """You are an experienced expert in reviewing clinical notes, skilled in comparing raw clinical note with agiven standardized template. Now, please compare the obtained raw clinical note with the givenstandardized clinical note template. The part that needs to be analyzed is the diseases list and differentialdiagnosis process. If you think this part is unreasonable, please give suggestions for improvement. , andoutput in JSON format as Dict{{"flag":false, "DDA_error": Str(suggestions for improvement)}}.

Below is the raw clinical note.
{raw_clinical_note}

Below is a standardized template for a standardized clinical note of the final diagnosis.
{standardized_clinical_note}"""
