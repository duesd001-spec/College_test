# app.py - Your SAT Question Generator Web App

import streamlit as st
import os # Make sure os is imported
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
# --- ADD THIS DEBUGGING BLOCK ---
st.write("### Debugging Secrets")
if 'GOOGLE_API_KEY' in st.secrets:
    st.success("API Key found in st.secrets!")
else:
    st.error("API Key not found in st.secrets.")
    # This will show you all the keys it *did* find, if any.
    st.write("Available secrets:", st.secrets.keys())
# --- END DEBUGGING BLOCK ---

# --- 1. SETUP AND CONFIGURATION ---

# Set your API key using Streamlit's secrets management
# This is more secure than putting the key directly in the code.
# You will set this up later when you deploy the app.
try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except:
    st.error("API Key not found. Please set it in your Streamlit secrets.")

# --- 2. SKILLS INSIGHT DATABASE ---
# (Your complete SKILLS_DATABASE dictionary goes here - I've trimmed it for brevity)
SKILLS_DATABASE = {
    "Reading And Writing": {
        "Information And Ideas": {
            1: "Students in this performance score band are beginning to obtain foundational skills to be college ready.",
            2: "Determine the most effective literary quotation to illustrate a straightforward claim about a character, setting, or theme; Locate relevant data points in informational graphics associated with passages at the middle grades level.",
            3: "Determine the most effective textual evidence (e.g., an additional finding; a quotation from a scholar) to support a claim in passages at the middle grades level as well as some at the high school level; Accurately identify explicitly stated and implicitly conveyed details in passages at the high school level.",
            4: "Determine the main idea of passages at the high school level; Make basic comparisons (e.g., determine highest/lowest value) among relevant data in informational graphics associated with passages at the middle grades level.",
            5: "Draw a reasonable text-based inference from passages at the middle grades level as well as some at the high school level; Make comparisons among relevant data in informational graphics associated with passages at the high school level in order to complete an example or illustrate or support a straightforward claim.",
            6: "Draw a reasonable text-based inference from passages at the high school level as well as some at the early college level; Determine the most effective literary quotation to support or illustrate an analytical claim about passages at the early college level; Interpret and integrate relevant data from informational graphics associated with passages at the high school level in order to support a claim.",
            7: "Draw a reasonable text-based inference from passages at the early college level; Determine the most effective textual evidence (e.g., a finding of a research study) to support or refute a claim in passages at the early college level; Interpret and integrate relevant data from informational graphics associated with passages at the early college level in order to support or refute a claim."
        },
        "Craft And Structure": {
            1: "Students in this performance score band are beginning to obtain foundational skills to be college ready.",
            2: "Determine the most logical and precise high-utility academic word or phrase to use in simple contexts and when the focal words and phrases are encountered frequently in texts at the middle grades level; Describe the function of a portion (e.g., a phrase or sentence) of a passage at the middle grades level in the context of the passage as a whole.",
            3: "Determine the most logical and precise high-utility academic word or phrase to use in moderately simple contexts and when the focal words and phrases are encountered frequently in texts at the middle grades level; Determine the meaning of a high-utility academic word or phrase in literary passages at the middle grades level; Describe the main purpose of passages at the middle grades level.",
            4: "Determine the most logical and precise high-utility academic word or phrase to use in moderately complex contexts and when the focal words and phrases are encountered frequently in texts at the high school level; Determine the meaning of a high-utility academic word or phrase, including the literal sense of a figurative word or phrase, in literary passages at the high school level; Describe the function of a portion (e.g., a phrase or sentence) of a passage at the high school level in the context of the passage as a whole.",
            5: "Determine the most logical and precise high-utility academic word or phrase to use in complex contexts and when the focal words and phrases are encountered frequently in texts at the high school level; Describe the main purpose of passages at the high school level when the authors’ goals are unstated; Draw a text-supported connection between two passages at the middle grades level on the same or similar topics.",
            6: "Determine the most logical and precise high-utility academic word or phrase to use in complex contexts and when the focal words and phrases are encountered frequently in texts at the early college level; Draw a text-supported connection between two passages at the high school level on the same or similar topics; Describe the function of a portion (e.g., a phrase or sentence) of a passage at the early college level in the context of the passage as a whole.",
            7: "Determine the most logical and precise high-utility academic word or phrase to use in highly complex contexts and when the focal words and phrases are encountered frequently in texts at the early college level; Draw a subtle text-supported connection between two passages at the early college level on the same or similar topics."
        },
        "Expression Of Ideas": {
            1: "Students in this performance score band are beginning to obtain foundational skills to be college ready.",
            2: "Determine the most effective transition word or phrase to introduce a supporting example (e.g., for instance); Determine the most effective transition word or phrase to indicate a logical relationship of time or sequence (e.g., later or next).",
            3: "Determine the most effective transition word or phrase to establish a logical relationship between two directly contrasting statements (e.g., however); Synthesize information from several statements to emphasize a similarity or difference.",
            4: "Determine the most effective transition word or phrase to indicate a cause-effect relationship between two statements (e.g., therefore); Synthesize information from several statements to emphasize a single feature or explain a concept.",
            5: "Determine the most effective transition word or phrase to signal a shift from a general discussion to a more specific case or example (e.g., specifically) or introduce a restatement of information (e.g., in short); Synthesize information from several complex statements to provide an explanation or form a comparison.",
            6: "Determine the most effective transition word or phrase to emphasize a point within a discussion (e.g., in fact); Synthesize information from several complex statements to provide a concise summary.",
            7: "Determine the most effective transition word or phrase to indicate an exception or counterpoint (e.g., granted); Synthesize information from several complex statements to make a rhetorically effective generalization."
        },
        "Standard English Conventions": {
            1: "Students in this performance score band are beginning to obtain foundational skills to be college ready.",
            2: "Maintain grammatical agreement between a subject and verb positioned closely together within a sentence; Determine correct verb formation in a fairly straightforward sentence.",
            3: "Maintain consistent verb tense in a sentence using two or more verbs in simple past or present tense; Determine when the possessive and/or plural form of a singular noun is required by the sense of a sentence; Maintain grammatical agreement between a subject pronoun and its singular referent.",
            4: "Use a comma to mark a boundary between a main clause and a supplementary phrase within a sentence; Use a period to punctuate the end of a declarative sentence, thereby avoiding creating a comma splice or run-on sentence; Maintain grammatical agreement between a noun and its pronoun in a straightforward sentence in which the pronoun precedes the referent.",
            5: "Use a semicolon to mark the boundary between two closely related independent clauses when the clauses are joined by a conjunctive adverb (e.g., however); Use commas to set off an interrupting nonessential sentence element.",
            6: "Eliminate unnecessary punctuation in challenging situations (e.g., between a long subject and the predicate or between two coordinate elements in a sentence); Use a colon to introduce an elaboration (e.g., a list of examples; a noun phrase renaming a previously mentioned concept); Use a period or semicolon to mark the boundary between two sentences when the boundary is subtle or requires careful reading to establish.",
            7: "Maintain grammatical agreement between a subject and verb in relatively complex sentences in which a substantial amount of text appears between the subject and main verb; Properly incorporate a restrictive sentence element, such as an appositive phrase modifying a noun phrase; Use a colon to introduce an independent clause elaborating on a statement or claim."
        }
    },
    "Math": {
        "Algebra": {
            1: "With or without a simple context, solve a one-step linear equation in one variable.",
            2: "Solve problems using a graph or linear equation when given one or more pieces of the following information: slope, intercepts, input-output pairs; Identify the coordinates of a solution, point, or intercept when given a graph of a linear equation or a graph of a system of two linear equations.",
            3: "With or without a simple context, create a linear equation in one variable or a system of two linear equations in two variables, and use the equation(s) to solve for an unknown value; Within a context, interpret the meaning of an input-output pair of a linear function.",
            4: "Within a complex context, choose the best interpretation of a part of an equation or of an input-output pair when given a linear equation that models the situation; Solve problems about linear relationships, making use of structure when present, that include equations, intercepts, slope, and input-output pairs, including finding equations for parallel and perpendicular lines.",
            5: "With or without a context, create a linear inequality in two variables or a system of two linear inequalities in two variables, and identify a point in the solution set; Within a context, create a linear function, and use it to find an unknown value; Make connections between an algebraic representation of a linear relationship and a graph or key features of the graph.",
            6: "Find and interpret the meaning of intercepts or slope for complex linear equations; Find the number of solutions to a complex linear equation; (SAT, PSAT/NMSQT, and PSAT 10 only) find the number of solutions to a system of two linear equations, or find missing coefficients of a linear equation or a system of two linear equations when the number of solutions is given; Make connections between a table, an algebraic representation, a graph, a solution, or features of a graph of a complex linear equation or a system of two linear equations.",
            7: "With or without a context, create and/or solve a linear equation or system of linear equations, or identify the correct coefficients or constants in the equation(s) that represent(s) the situation; Make connections between different representations of linear equations in one variable, linear functions, linear equations in two variables, systems of two linear equations in two variables, and (SAT, PSAT/NMSQT, and PSAT 10 only) linear inequalities when these representations include symbolic representations that may contain variable constants."
        },
        "Advanced Math": {
            1: "Students in this performance score band are beginning to obtain foundational skills to be college ready.",
            2: "Identify a key feature of a graph, such as an intercept, a solution, or (SAT, PSAT/NMSQT, and PSAT 10 only) a translation, when given the graph of either a nonlinear function or a system consisting of a linear and a nonlinear function; Rewrite an expression by combining like terms, factoring out a greatest common factor, or applying the distributive property.",
            3: "With or without a context, use a given quadratic or exponential equation that represents the relationship between two variables to find an unknown value; Evaluate a function at a given value, or solve for the input when the output is given; (PSAT 8/9 only) solve polynomial equations by factoring out a greatest common factor.",
            4: "With or without a context, use a quadratic or exponential equation that represents the relationship between two variables, or (SAT, PSAT/NMSQT, and PSAT 10 only) create and use a quadratic or exponential equation that represents the relationship between two variables; Solve quadratic equations using factoring; solve equations that include radical or rational terms, or solve a system of one linear and one nonlinear equation; (PSAT/NMSQT and PSAT 10 only) solve polynomial equations using factoring.",
            5: "Within a context, interpret the meaning of a constant or a variable in an exponential or quadratic equation; Make connections between a graph of a nonlinear function and its equation, and identify key features of the graph; (SAT, PSAT/NMSQT, and PSAT 10 only) add, subtract, and multiply polynomials.",
            6: "(SAT, PSAT/NMSQT, and PSAT 10 only) Within a context, create a quadratic or exponential equation that represents the situation, and solve for an unknown value; (SAT, PSAT/NMSQT, and PSAT 10 only) Within a context, interpret a key feature of the graph of an exponential or quadratic equation representing the situation; Make connections between either the graph of a quadratic or exponential function or points on the graph of a function and its algebraic representation, and (SAT, PSAT/NMSQT, and PSAT 10 only) understand how a translation of a function affects the graph or the equation.",
            7: "Solve problems with or without context involving one or more nonlinear equations to find the value of an unknown constant; Solve a complex equation or formula for a variable of interest; (SAT only) use properties of exponents and properties of polynomial, rational, and radical expressions to rewrite complex expressions, using structure when present, or determine the most suitable form of an equation to display a certain feature."
        },
        "Problem-Solving And Data Analysis": {
            1: "Solve simple problems using percents or unit rates; Find the median of a list of values presented in ascending order.",
            2: "Solve problems using percentages, unit rates, and unit conversions; Read, compare, and interpret data presented in a bar graph or frequency table.",
            3: "Solve problems involving percent, including finding percentages and solving problems in which the percentage is greater than 100; Read and interpret data displayed in a two-way table; calculate the probability of an event from a frequency table or a two-way table.",
            4: "Identify, interpret, and use ratios, proportions, percentages, and rates, expressing them in equivalent forms, to solve problems; With or without a context, compare and contrast data sets using mean, median, and (SAT, PSAT/NMSQT, and PSAT 10 only) standard deviation; Fit a linear model to data displayed in a scatterplot; (SAT only) Select plausible values of the population mean or population proportion when given a sample mean or sample proportion, respectively, and the associated margin of error.",
            5: "With or without a context, solve problems using growth factor expressed as a percent or complex unit; Apply the understanding that the probability of all possible outcomes of an event has a sum of 1.",
            6: "Solve multistep problems using ratios, rates, percentages, and derived units, including problems that arise from products and quotients; Determine the a verage rate of change for data displayed in a graph; determine the mean and median of a data set presented in a frequency table; Calculate the conditional probability of an event from a two-way table.",
            7: "(SAT only) Identify or describe the population to which the results of a research study can be extended; Determine how the mean, median, and range of a data set are affected by changes in the data set."
        },
        "Geometry And Trigonometry": {
            1: "Find the volume of a right rectangular prism when given the lengths of the edges and the formula for the volume; Find the area of a rectangle when given the lengths of the sides.",
            2: "Solve problems involving the perimeter and side lengths of plane figures; (SAT, PSAT/NMSQT, and PSAT 10 only) Solve problems by applying theorems related to parallel lines cut by a transversal.",
            3: "Solve problems involving the area and side lengths of plane figures; Find the measure of an angle by applying definitions and theorems about angles, such as the triangle angle sum theorem and (SAT, PSAT/NMSQT, and PSAT 10 only) theorems related to angles formed by intersecting lines; Use the Pythagorean theorem to find the length of a hypotenuse in a right triangle when given the lengths of the two legs.",
            4: "Solve problems involving the area of a plane figure or the volume of a cube or pyramid; Solve problems using concepts and theorems related to scale factors, the sum of angles of triangles, and (SAT, PSAT/NMSQT, and PSAT 10 only) congruence and similarity; Find a side length in a given triangle by applying the Pythagorean theorem; (PSAT/NMSQT and PSAT 10 only) find an angle measure in or a side length of a given triangle using the properties of special right triangles.",
            5: "(SAT only) Solve problems using the relationship between sine and cosine of complementary angles; (SAT only) convert between degree measure and radian measure; (SAT only) Write an equation of a circle in the xy-plane when given the center of the circle and a point that lies on the circle; Solve multistep problems involving area and perimeter of plane figures.",
            6: "Calculate the surface area, the volume, or a dimension of a prism when given other information about the prism; Solve complex problems by applying the Pythagorean theorem to find a side length of a rectangle or by applying the triangle angle sum theorem; (SAT, PSAT/NMSQT, and PSAT 10 only) Solve complex problems by using concepts and theorems related to congruence and similarity of right triangles, including, but not limited to, trigonometric ratios of right triangles, or by identifying the impact of changes by a scale factor on perimeter, area, and volume.",
            7: "Solve for missing values in objects modeled by various 2D and 3D geometric shapes by applying formulas for area, surface area, or volume; (SAT, PSAT/NMSQT, and PSAT 10 only) Solve complex problems by applying properties of similar and congruent triangles, theorems related to angles and triangles, or right triangle trigonometry, or (SAT only) use similarity to calculate values of trigonometric ratios; (SAT only) Solve problems using properties and theorems related to circles and parts of circles, such as radii, diameters, tangents, angles, arcs, arc length, and sectors."
        }
    }
}

# --- 3. THE AI GENERATOR FUNCTION ---
# (This is the same function you perfected earlier)
def generate_sat_question(original_text, subtest, domain, score_band):
    """
    Generates a leveled SAT question using a Large Language Model.
    This new version asks the LLM to first level the text and then create the question.
    """
    # 1. Get the target skill from the database
    try:
        target_skill = SKILLS_DATABASE[subtest][domain][score_band]
    except KeyError:
        return "Error: The selected subtest, domain, or score band is not in the database."

    # We no longer need the old level_text_for_score_band function.
    # The LLM will handle this.

    # 2. Set up the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

    # 3. Create a new, more powerful two-step prompt for the LLM
    prompt_template = """
    You are an expert SAT tutor and content creator. Your task is to perform two steps:
    First, rewrite a user-provided text to match a specific SAT score band's complexity.
    Second, use that newly rewritten text to create a high-quality, multiple-choice question.

    **Step 1: Rewrite the Text**
    - Analyze the original text provided below.
    - Rewrite it so its vocabulary, sentence structure, and complexity are appropriate for a student in **Score Band {score_band}**.
    - For lower score bands (1-3), use simpler language and shorter sentences.
    - For higher score bands (5-7), use more sophisticated vocabulary and more complex sentence structures.
    - Present the result under the heading "Leveled Text:".

    **Step 2: Generate a Question from the Leveled Text**
    - Using ONLY the "Leveled Text" you just created, generate one multiple-choice question.
    - The question must specifically assess this skill: **"{skill}"**
    - The question should be appropriate for the **{subtest}** section's **{domain}** domain.
    - Present the result under the headings "Question:", "Choices:", and "Feedback:".
    - The feedback must explain the correct answer and why the others are wrong, referencing the Leveled Text.

    **Original Text:**
    ---
    {text}
    ---
    """
    prompt = PromptTemplate(
        input_variables=["text", "subtest", "domain", "score_band", "skill"],
        template=prompt_template,
    )

    # 4. Create and run the LLM chain
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run({
        "text": original_text,
        "subtest": subtest,
        "domain": domain,
        "score_band": score_band,
        "skill": target_skill,
    })

    return response


# --- 4. STREAMLIT WEB INTERFACE ---

st.title("🤖 AI-Powered SAT Question Generator")
st.markdown("This tool uses AI to create leveled SAT questions based on your text and specifications.")

# Create columns for a cleaner layout
col1, col2, col3 = st.columns(3)

with col1:
    subtest = st.selectbox("Select Subtest:", options=list(SKILLS_DATABASE.keys()))

with col2:
    # The domain options depend on the selected subtest
    domain = st.selectbox("Select Content Domain:", options=list(SKILLS_DATABASE[subtest].keys()))

with col3:
    score_band = st.selectbox("Select Score Band:", options=list(range(1, 8)))

# Text area for user input
uploaded_text = st.text_area("Paste your text here:", height=250)

# Generate button
if st.button("Generate Question"):
    if uploaded_text:
        with st.spinner("The AI is thinking... 🧠"):
            # Call your generator function with the user's inputs
            response = generate_sat_question(uploaded_text, subtest, domain, score_band)
            st.markdown("---")
            st.header("Generated Output")
            st.markdown(response)
    else:
        st.warning("Please paste some text to generate a question.")