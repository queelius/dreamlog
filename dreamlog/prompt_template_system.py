"""
Prompt Template System for DreamLog

A flexible, parameterized prompt template system that can be tuned per LLM model.
Supports categories of prompts and learns what works best for different query types.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random
import json
from pathlib import Path


# Example database for few-shot learning
RULE_EXAMPLES = [
    # Family relationships
    {
        "domain": "family",
        "prolog": "grandparent(X, Z) :- parent(X, Y), parent(Y, Z).",
        "json": [["rule", ["grandparent", "X", "Z"], [["parent", "X", "Y"], ["parent", "Y", "Z"]]]]
    },
    {
        "domain": "family",
        "prolog": "sibling(X, Y) :- parent(Z, X), parent(Z, Y).",
        "json": [["rule", ["sibling", "X", "Y"], [["parent", "Z", "X"], ["parent", "Z", "Y"]]]]
    },
    {
        "domain": "family",
        "prolog": "uncle(X, Y) :- parent(Z, Y), sibling(X, Z).",
        "json": [["rule", ["uncle", "X", "Y"], [["parent", "Z", "Y"], ["sibling", "X", "Z"]]]]
    },
    {
        "domain": "family",
        "prolog": "cousin(X, Y) :- parent(A, X), parent(B, Y), sibling(A, B).",
        "json": [["rule", ["cousin", "X", "Y"], [["parent", "A", "X"], ["parent", "B", "Y"], ["sibling", "A", "B"]]]]
    },
    {
        "domain": "family",
        "prolog": "ancestor(X, Y) :- parent(X, Y).",
        "json": [["rule", ["ancestor", "X", "Y"], [["parent", "X", "Y"]]]]
    },
    # Geography
    {
        "domain": "geography",
        "prolog": "in_continent(City, Continent) :- in_country(City, Country), country_in(Country, Continent).",
        "json": [["rule", ["in_continent", "City", "Continent"], [["in_country", "City", "Country"], ["country_in", "Country", "Continent"]]]]
    },
    {
        "domain": "geography",
        "prolog": "neighbor_cities(X, Y) :- in_country(X, C), in_country(Y, C), adjacent(X, Y).",
        "json": [["rule", ["neighbor_cities", "X", "Y"], [["in_country", "X", "C"], ["in_country", "Y", "C"], ["adjacent", "X", "Y"]]]]
    },
    # Programming
    {
        "domain": "programming",
        "prolog": "web_framework(F, L) :- uses(L, F), framework_type(F, web).",
        "json": [["rule", ["web_framework", "F", "L"], [["uses", "L", "F"], ["framework_type", "F", "web"]]]]
    },
    {
        "domain": "programming",
        "prolog": "compiled_language(L) :- language(L, compiled).",
        "json": [["rule", ["compiled_language", "L"], [["language", "L", "compiled"]]]]
    },
    # Academic
    {
        "domain": "academic",
        "prolog": "advisor_chain(X, Z) :- advises(X, Y), advises(Y, Z).",
        "json": [["rule", ["advisor_chain", "X", "Z"], [["advises", "X", "Y"], ["advises", "Y", "Z"]]]]
    },
    {
        "domain": "academic",
        "prolog": "coauthor(X, Y) :- wrote(X, P), wrote(Y, P).",
        "json": [["rule", ["coauthor", "X", "Y"], [["wrote", "X", "P"], ["wrote", "Y", "P"]]]]
    },
    # Organization
    {
        "domain": "organization",
        "prolog": "manager_chain(X, Z) :- manages(X, Y), manages(Y, Z).",
        "json": [["rule", ["manager_chain", "X", "Z"], [["manages", "X", "Y"], ["manages", "Y", "Z"]]]]
    },
    {
        "domain": "organization",
        "prolog": "colleague(X, Y) :- works_in(X, D), works_in(Y, D).",
        "json": [["rule", ["colleague", "X", "Y"], [["works_in", "X", "D"], ["works_in", "Y", "D"]]]]
    },
    # Graph/Network
    {
        "domain": "graph",
        "prolog": "path(X, Y) :- edge(X, Y).",
        "json": [["rule", ["path", "X", "Y"], [["edge", "X", "Y"]]]]
    },
    {
        "domain": "graph",
        "prolog": "reachable(X, Z) :- edge(X, Y), path(Y, Z).",
        "json": [["rule", ["reachable", "X", "Z"], [["edge", "X", "Y"], ["path", "Y", "Z"]]]]
    },
    {
        "domain": "graph",
        "prolog": "connected(X, Y) :- reachable(X, Y).",
        "json": [["rule", ["connected", "X", "Y"], [["reachable", "X", "Y"]]]]
    },
    # Medical/Health
    {
        "domain": "medical",
        "prolog": "treatment_for(Drug, Disease) :- targets(Drug, Symptom), causes(Disease, Symptom).",
        "json": [["rule", ["treatment_for", "Drug", "Disease"], [["targets", "Drug", "Symptom"], ["causes", "Disease", "Symptom"]]]]
    },
    {
        "domain": "medical",
        "prolog": "contraindicated(Drug, Patient) :- allergic_to(Patient, Drug).",
        "json": [["rule", ["contraindicated", "Drug", "Patient"], [["allergic_to", "Patient", "Drug"]]]]
    },
    {
        "domain": "medical",
        "prolog": "high_risk(Patient, Disease) :- has_symptom(Patient, S), risk_factor(S, Disease).",
        "json": [["rule", ["high_risk", "Patient", "Disease"], [["has_symptom", "Patient", "S"], ["risk_factor", "S", "Disease"]]]]
    },
    # Commerce/Business
    {
        "domain": "commerce",
        "prolog": "can_buy(Customer, Product) :- has_funds(Customer, Amount), price(Product, Cost), greater_equal(Amount, Cost).",
        "json": [["rule", ["can_buy", "Customer", "Product"], [["has_funds", "Customer", "Amount"], ["price", "Product", "Cost"], ["greater_equal", "Amount", "Cost"]]]]
    },
    {
        "domain": "commerce",
        "prolog": "supplier_of(Company, Product) :- manufactures(Company, Product).",
        "json": [["rule", ["supplier_of", "Company", "Product"], [["manufactures", "Company", "Product"]]]]
    },
    {
        "domain": "commerce",
        "prolog": "competitor(X, Y) :- sells(X, P), sells(Y, P), different(X, Y).",
        "json": [["rule", ["competitor", "X", "Y"], [["sells", "X", "P"], ["sells", "Y", "P"], ["different", "X", "Y"]]]]
    },
    # Transportation
    {
        "domain": "transportation",
        "prolog": "can_reach(From, To) :- route(From, To).",
        "json": [["rule", ["can_reach", "From", "To"], [["route", "From", "To"]]]]
    },
    {
        "domain": "transportation",
        "prolog": "transfer_route(From, To) :- route(From, Hub), route(Hub, To).",
        "json": [["rule", ["transfer_route", "From", "To"], [["route", "From", "Hub"], ["route", "Hub", "To"]]]]
    },
    {
        "domain": "transportation",
        "prolog": "same_line(Stop1, Stop2) :- on_line(Stop1, Line), on_line(Stop2, Line).",
        "json": [["rule", ["same_line", "Stop1", "Stop2"], [["on_line", "Stop1", "Line"], ["on_line", "Stop2", "Line"]]]]
    },
    # Food/Cuisine
    {
        "domain": "food",
        "prolog": "can_make(Dish, Chef) :- requires(Dish, Ingredient), has_ingredient(Chef, Ingredient).",
        "json": [["rule", ["can_make", "Dish", "Chef"], [["requires", "Dish", "Ingredient"], ["has_ingredient", "Chef", "Ingredient"]]]]
    },
    {
        "domain": "food",
        "prolog": "vegetarian_dish(Dish) :- ingredient_in(I, Dish), vegetarian(I).",
        "json": [["rule", ["vegetarian_dish", "Dish"], [["ingredient_in", "I", "Dish"], ["vegetarian", "I"]]]]
    },
    {
        "domain": "food",
        "prolog": "pairs_with(Food, Wine) :- flavor(Food, F), complements(Wine, F).",
        "json": [["rule", ["pairs_with", "Food", "Wine"], [["flavor", "Food", "F"], ["complements", "Wine", "F"]]]]
    },
    # Sports
    {
        "domain": "sports",
        "prolog": "team_rivals(T1, T2) :- in_division(T1, D), in_division(T2, D), different(T1, T2).",
        "json": [["rule", ["team_rivals", "T1", "T2"], [["in_division", "T1", "D"], ["in_division", "T2", "D"], ["different", "T1", "T2"]]]]
    },
    {
        "domain": "sports",
        "prolog": "championship_eligible(Team) :- wins(Team, W), greater_equal(W, 10).",
        "json": [["rule", ["championship_eligible", "Team"], [["wins", "Team", "W"], ["greater_equal", "W", 10]]]]
    },
    {
        "domain": "sports",
        "prolog": "teammates(P1, P2) :- plays_for(P1, Team), plays_for(P2, Team), different(P1, P2).",
        "json": [["rule", ["teammates", "P1", "P2"], [["plays_for", "P1", "Team"], ["plays_for", "P2", "Team"], ["different", "P1", "P2"]]]]
    },
    # Science/Chemistry
    {
        "domain": "science",
        "prolog": "forms_compound(E1, E2, Compound) :- element(E1), element(E2), bonds_with(E1, E2, Compound).",
        "json": [["rule", ["forms_compound", "E1", "E2", "Compound"], [["element", "E1"], ["element", "E2"], ["bonds_with", "E1", "E2", "Compound"]]]]
    },
    {
        "domain": "science",
        "prolog": "soluble_in(Substance, Solvent) :- polarity(Substance, P), polarity(Solvent, P).",
        "json": [["rule", ["soluble_in", "Substance", "Solvent"], [["polarity", "Substance", "P"], ["polarity", "Solvent", "P"]]]]
    },
    # Education
    {
        "domain": "education",
        "prolog": "prerequisite_chain(C1, C3) :- prerequisite(C1, C2), prerequisite(C2, C3).",
        "json": [["rule", ["prerequisite_chain", "C1", "C3"], [["prerequisite", "C1", "C2"], ["prerequisite", "C2", "C3"]]]]
    },
    {
        "domain": "education",
        "prolog": "can_enroll(Student, Course) :- completed(Student, Prereq), prerequisite(Prereq, Course).",
        "json": [["rule", ["can_enroll", "Student", "Course"], [["completed", "Student", "Prereq"], ["prerequisite", "Prereq", "Course"]]]]
    },
    {
        "domain": "education",
        "prolog": "classmates(S1, S2) :- enrolled_in(S1, C), enrolled_in(S2, C), different(S1, S2).",
        "json": [["rule", ["classmates", "S1", "S2"], [["enrolled_in", "S1", "C"], ["enrolled_in", "S2", "C"], ["different", "S1", "S2"]]]]
    },
    # Library/Books
    {
        "domain": "library",
        "prolog": "available(Book) :- in_library(Book), not_checked_out(Book).",
        "json": [["rule", ["available", "Book"], [["in_library", "Book"], ["not_checked_out", "Book"]]]]
    },
    {
        "domain": "library",
        "prolog": "same_author(B1, B2) :- written_by(B1, A), written_by(B2, A), different(B1, B2).",
        "json": [["rule", ["same_author", "B1", "B2"], [["written_by", "B1", "A"], ["written_by", "B2", "A"], ["different", "B1", "B2"]]]]
    },
    {
        "domain": "library",
        "prolog": "recommended_for(Book, Reader) :- genre(Book, G), likes(Reader, G).",
        "json": [["rule", ["recommended_for", "Book", "Reader"], [["genre", "Book", "G"], ["likes", "Reader", "G"]]]]
    },
    # Movies/Entertainment
    {
        "domain": "movies",
        "prolog": "acted_together(A1, A2) :- acts_in(A1, M), acts_in(A2, M), different(A1, A2).",
        "json": [["rule", ["acted_together", "A1", "A2"], [["acts_in", "A1", "M"], ["acts_in", "A2", "M"], ["different", "A1", "A2"]]]]
    },
    {
        "domain": "movies",
        "prolog": "directed_actor(Director, Actor) :- directed(Director, Movie), acts_in(Actor, Movie).",
        "json": [["rule", ["directed_actor", "Director", "Actor"], [["directed", "Director", "Movie"], ["acts_in", "Actor", "Movie"]]]]
    },
    {
        "domain": "movies",
        "prolog": "similar_movies(M1, M2) :- genre(M1, G), genre(M2, G), different(M1, M2).",
        "json": [["rule", ["similar_movies", "M1", "M2"], [["genre", "M1", "G"], ["genre", "M2", "G"], ["different", "M1", "M2"]]]]
    },
    # Music
    {
        "domain": "music",
        "prolog": "band_member(Person, Band) :- plays_in(Person, Band).",
        "json": [["rule", ["band_member", "Person", "Band"], [["plays_in", "Person", "Band"]]]]
    },
    {
        "domain": "music",
        "prolog": "collaborated(M1, M2) :- performed_on(M1, Album), performed_on(M2, Album), different(M1, M2).",
        "json": [["rule", ["collaborated", "M1", "M2"], [["performed_on", "M1", "Album"], ["performed_on", "M2", "Album"], ["different", "M1", "M2"]]]]
    },
    {
        "domain": "music",
        "prolog": "influenced_by(Artist, Influence) :- style(Artist, S), pioneered(Influence, S).",
        "json": [["rule", ["influenced_by", "Artist", "Influence"], [["style", "Artist", "S"], ["pioneered", "Influence", "S"]]]]
    },
    # Social Networks
    {
        "domain": "social",
        "prolog": "friend_of_friend(X, Z) :- friend(X, Y), friend(Y, Z), different(X, Z).",
        "json": [["rule", ["friend_of_friend", "X", "Z"], [["friend", "X", "Y"], ["friend", "Y", "Z"], ["different", "X", "Z"]]]]
    },
    {
        "domain": "social",
        "prolog": "mutual_friend(X, Y, Z) :- friend(X, Z), friend(Y, Z), different(X, Y).",
        "json": [["rule", ["mutual_friend", "X", "Y", "Z"], [["friend", "X", "Z"], ["friend", "Y", "Z"], ["different", "X", "Y"]]]]
    },
    {
        "domain": "social",
        "prolog": "influencer(Person) :- follower(F, Person), count(F, N), greater(N, 1000).",
        "json": [["rule", ["influencer", "Person"], [["follower", "F", "Person"], ["count", "F", "N"], ["greater", "N", 1000]]]]
    },
    # Law/Legal
    {
        "domain": "legal",
        "prolog": "precedent_applies(Case, Law) :- similar_facts(Case, PriorCase), ruled_by(PriorCase, Law).",
        "json": [["rule", ["precedent_applies", "Case", "Law"], [["similar_facts", "Case", "PriorCase"], ["ruled_by", "PriorCase", "Law"]]]]
    },
    {
        "domain": "legal",
        "prolog": "conflict_of_interest(Lawyer, Case) :- represents(Lawyer, Party1), opposes(Party1, Party2), related_to(Lawyer, Party2).",
        "json": [["rule", ["conflict_of_interest", "Lawyer", "Case"], [["represents", "Lawyer", "Party1"], ["opposes", "Party1", "Party2"], ["related_to", "Lawyer", "Party2"]]]]
    },
    # Real Estate
    {
        "domain": "real_estate",
        "prolog": "neighborhood(H1, H2) :- address(H1, Street, City), address(H2, Street, City), different(H1, H2).",
        "json": [["rule", ["neighborhood", "H1", "H2"], [["address", "H1", "Street", "City"], ["address", "H2", "Street", "City"], ["different", "H1", "H2"]]]]
    },
    {
        "domain": "real_estate",
        "prolog": "affordable(Property, Buyer) :- price(Property, P), budget(Buyer, B), less_equal(P, B).",
        "json": [["rule", ["affordable", "Property", "Buyer"], [["price", "Property", "P"], ["budget", "Buyer", "B"], ["less_equal", "P", "B"]]]]
    },
    # Biology/Ecology
    {
        "domain": "biology",
        "prolog": "predator_of(X, Y) :- eats(X, Y), animal(X), animal(Y).",
        "json": [["rule", ["predator_of", "X", "Y"], [["eats", "X", "Y"], ["animal", "X"], ["animal", "Y"]]]]
    },
    {
        "domain": "biology",
        "prolog": "food_chain(X, Z) :- eats(X, Y), eats(Y, Z).",
        "json": [["rule", ["food_chain", "X", "Z"], [["eats", "X", "Y"], ["eats", "Y", "Z"]]]]
    },
    {
        "domain": "biology",
        "prolog": "ecosystem_member(Organism, Ecosystem) :- lives_in(Organism, Habitat), part_of(Habitat, Ecosystem).",
        "json": [["rule", ["ecosystem_member", "Organism", "Ecosystem"], [["lives_in", "Organism", "Habitat"], ["part_of", "Habitat", "Ecosystem"]]]]
    },
    # Finance
    {
        "domain": "finance",
        "prolog": "investment_risk(Stock, Level) :- volatility(Stock, V), classify_risk(V, Level).",
        "json": [["rule", ["investment_risk", "Stock", "Level"], [["volatility", "Stock", "V"], ["classify_risk", "V", "Level"]]]]
    },
    {
        "domain": "finance",
        "prolog": "diversified(Portfolio) :- holds(Portfolio, A1), holds(Portfolio, A2), different_sector(A1, A2).",
        "json": [["rule", ["diversified", "Portfolio"], [["holds", "Portfolio", "A1"], ["holds", "Portfolio", "A2"], ["different_sector", "A1", "A2"]]]]
    },
    {
        "domain": "finance",
        "prolog": "profitable_trade(Buy, Sell) :- bought_at(Buy, P1), sold_at(Sell, P2), greater(P2, P1).",
        "json": [["rule", ["profitable_trade", "Buy", "Sell"], [["bought_at", "Buy", "P1"], ["sold_at", "Sell", "P2"], ["greater", "P2", "P1"]]]]
    },
    # Weather/Climate
    {
        "domain": "weather",
        "prolog": "similar_climate(City1, City2) :- avg_temp(City1, T), avg_temp(City2, T), different(City1, City2).",
        "json": [["rule", ["similar_climate", "City1", "City2"], [["avg_temp", "City1", "T"], ["avg_temp", "City2", "T"], ["different", "City1", "City2"]]]]
    },
    {
        "domain": "weather",
        "prolog": "rainy_season(Location, Month) :- precipitation(Location, Month, P), high_rainfall(P).",
        "json": [["rule", ["rainy_season", "Location", "Month"], [["precipitation", "Location", "Month", "P"], ["high_rainfall", "P"]]]]
    },
    # Gaming
    {
        "domain": "gaming",
        "prolog": "can_defeat(Player, Enemy) :- level(Player, L1), level(Enemy, L2), greater(L1, L2).",
        "json": [["rule", ["can_defeat", "Player", "Enemy"], [["level", "Player", "L1"], ["level", "Enemy", "L2"], ["greater", "L1", "L2"]]]]
    },
    {
        "domain": "gaming",
        "prolog": "unlocked(Achievement, Player) :- completed(Player, Quest), grants(Quest, Achievement).",
        "json": [["rule", ["unlocked", "Achievement", "Player"], [["completed", "Player", "Quest"], ["grants", "Quest", "Achievement"]]]]
    },
    {
        "domain": "gaming",
        "prolog": "team_game(G1, G2) :- multiplayer(G1), multiplayer(G2), same_genre(G1, G2).",
        "json": [["rule", ["team_game", "G1", "G2"], [["multiplayer", "G1"], ["multiplayer", "G2"], ["same_genre", "G1", "G2"]]]]
    },
    # History
    {
        "domain": "history",
        "prolog": "contemporary(P1, P2) :- lived_during(P1, Era), lived_during(P2, Era), different(P1, P2).",
        "json": [["rule", ["contemporary", "P1", "P2"], [["lived_during", "P1", "Era"], ["lived_during", "P2", "Era"], ["different", "P1", "P2"]]]]
    },
    {
        "domain": "history",
        "prolog": "influenced_event(Person, Event) :- participated_in(Person, E1), led_to(E1, Event).",
        "json": [["rule", ["influenced_event", "Person", "Event"], [["participated_in", "Person", "E1"], ["led_to", "E1", "Event"]]]]
    },
]


def sample_examples(num_examples: int = 5, seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Sample examples from the database without replacement.

    Args:
        num_examples: Number of examples to sample (default 5)
        seed: Random seed for reproducibility (optional)

    Returns:
        List of sampled example dictionaries
    """
    if seed is not None:
        random.seed(seed)

    # Sample without replacement
    num_to_sample = min(num_examples, len(RULE_EXAMPLES))
    return random.sample(RULE_EXAMPLES, num_to_sample)


class PromptCategory(Enum):
    """Categories of prompt templates for different reasoning tasks"""
    COMPRESSION = "compression"
    ABSTRACTION = "abstraction"
    ANALOGY = "analogy"
    COUNTERFACTUAL = "counterfactual"
    DECOMPOSITION = "decomposition"
    BRIDGE = "bridge"
    DEFINITION = "definition"  # For undefined predicates
    EXAMPLE_GENERATION = "example_generation"
    CONSOLIDATION = "consolidation"  # For consolidating knowledge


@dataclass
class QueryContext:
    """Context for a query to help select appropriate template"""
    term: Any  # JSON format: ["functor", "arg1", ...]
    kb_facts: List[Any] = field(default_factory=list)  # JSON format facts
    kb_rules: List[Tuple[Any, List[Any]]] = field(default_factory=list)  # JSON format rules
    existing_functors: List[str] = field(default_factory=list)
    
    @property
    def is_empty_kb(self) -> bool:
        return len(self.kb_facts) == 0 and len(self.kb_rules) == 0
    
    @property
    def has_examples(self) -> bool:
        return len(self.kb_facts) > 0
    
    @property
    def has_rules(self) -> bool:
        return len(self.kb_rules) > 0


@dataclass
class ModelParameters:
    """Parameters that can be tuned per LLM model"""
    model_name: str
    
    # Many-shot learning parameters
    min_examples: int = 3
    max_examples: int = 8
    optimal_examples: int = 5  # Learned over time
    
    # Context window management
    max_context_tokens: int = 4000
    example_selection_strategy: str = "similarity"  # similarity, diversity, mixed
    
    # Response format preferences
    prefers_json: bool = True
    needs_explicit_format: bool = True
    handles_complex_reasoning: bool = True
    
    # Temperature for different tasks
    temperature_by_category: Dict[str, float] = field(default_factory=lambda: {
        "compression": 0.1,
        "abstraction": 0.3,
        "analogy": 0.5,
        "counterfactual": 0.7,
        "decomposition": 0.2,
        "bridge": 0.4,
        "definition": 0.1,
        "example_generation": 0.3
    })
    
    # Success rates by category (learned)
    success_rates: Dict[str, float] = field(default_factory=dict)
    
    # Prompt style preferences (learned)
    prompt_style_scores: Dict[str, float] = field(default_factory=lambda: {
        "verbose": 0.5,
        "concise": 0.5,
        "step_by_step": 0.5,
        "direct": 0.5
    })


@dataclass
class PromptTemplate:
    """A single prompt template"""
    id: str
    category: PromptCategory
    template: str
    variables: List[str]  # Variables to fill in
    
    # Performance tracking
    use_count: int = 0
    success_count: int = 0
    avg_response_quality: float = 0.0
    
    # Model-specific success rates
    model_success_rates: Dict[str, float] = field(default_factory=dict)
    
    def render(self, **kwargs) -> str:
        """Render the template with the provided variables"""
        result = self.template
        for var in self.variables:
            if var in kwargs:
                # Simple replacement - could be more sophisticated
                result = result.replace(f"{{{var}}}", str(kwargs[var]))
        return result
    
    @property
    def success_rate(self) -> float:
        if self.use_count == 0:
            return 0.0
        return self.success_count / self.use_count


class PromptTemplateLibrary:
    """Library of prompt templates organized by category"""

    def __init__(self, model_name: str = "unknown", example_retriever=None):
        self.model_name = model_name
        self.model_params = ModelParameters(model_name=model_name)
        self.templates: Dict[PromptCategory, List[PromptTemplate]] = {
            category: [] for category in PromptCategory
        }
        self.example_retriever = example_retriever
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize with starter templates for each category"""
        
        # COMPRESSION templates
        self.templates[PromptCategory.COMPRESSION].extend([
            PromptTemplate(
                id="compress_patterns",
                category=PromptCategory.COMPRESSION,
                template="""Find patterns in these facts that could be expressed as a single rule:
{facts}

Return the compressed rule in S-expression format:
(rule (head args...) ((body1 args...) (body2 args...)))""",
                variables=["facts"]
            ),
            PromptTemplate(
                id="compress_redundant",
                category=PromptCategory.COMPRESSION,
                template="""These rules seem redundant:
{rules}

Can you merge them into a more general rule?
Output format: (rule (head args...) ((body1 args...) (body2 args...)))""",
                variables=["rules"]
            )
        ])
        
        # DEFINITION templates (for undefined predicates)
        self.templates[PromptCategory.DEFINITION].extend([
            PromptTemplate(
                id="define_simple",
                category=PromptCategory.DEFINITION,
                template="""Query: {query}
Context: {context}

Define the predicate '{functor}' with {arity} arguments.

S-expression format (return ONLY S-expressions, one per line):
- Fact: (parent alice bob)
- Rule: (rule (grandparent X Z) ((parent X Y) (parent Y Z)))
- Rule with multiple conditions: (rule (sibling X Y) ((parent Z X) (parent Z Y) (different X Y)))

Your output (S-expressions only, one per line):""",
                variables=["query", "context", "functor", "arity"]
            ),
            PromptTemplate(
                id="define_with_examples",
                category=PromptCategory.DEFINITION,
                template="""You are a logic programming expert helping define predicates in Prolog.

## Your Query
{query}

## Current Knowledge Base
{context}

## Your Task
Define the predicate '{functor}' to answer the query above.

---

## FEW-SHOT EXAMPLES

Here are complete examples showing how to reason about queries:

### Example 1: Query about gender (primitive property)
**Query:** `male(john)`
**KB Facts:** `parent(john, mary). parent(john, tom).`
**Reasoning:** The predicate `male` is a primitive property - it cannot be derived from `parent` relationships. The name "john" is typically male. I should generate a fact.
**Output:**
```prolog
male(john).
```

### Example 2: Query about gender with female name
**Query:** `female(mary)`
**KB Facts:** `parent(john, mary). parent(mary, alice).`
**Reasoning:** The predicate `female` is a primitive property. The name "mary" is typically female. I should generate a fact.
**Output:**
```prolog
female(mary).
```

### Example 3: Query about a derivable relationship
**Query:** `father(X, Y)`
**KB Facts:** `parent(john, mary). male(john).`
**Reasoning:** A father is a male parent. I have both `parent` and `male` predicates in the KB. I should generate a rule.
**Output:**
```prolog
father(X, Y) :- parent(X, Y), male(X).
```

### Example 4: Query about ancestry (recursive)
**Query:** `ancestor(X, Y)`
**KB Facts:** `parent(john, mary). parent(mary, alice).`
**Reasoning:** An ancestor is either a parent (base case) or a parent of an ancestor (recursive case). I should generate rules.
**Output:**
```prolog
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
```

### Example 5: Query about grandparent
**Query:** `grandparent(X, Z)`
**KB Facts:** `parent(john, mary). parent(mary, alice).`
**Reasoning:** A grandparent is a parent of a parent. I can derive this from the `parent` predicate.
**Output:**
```prolog
grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
```

### Example 6: Negative inference from name
**Query:** `male(mary)`
**KB Facts:** `parent(mary, alice). parent(john, mary).`
**Reasoning:** The query asks if mary is male. The name "mary" is typically female, not male. I should NOT generate `male(mary)`. Instead, I can generate the opposite fact if useful, or generate nothing for male(mary).
**Output:**
```prolog
female(mary).
```

---

## DECISION RULES

1. **Primitive properties (use your world knowledge):**
   - For predicates like `male/female`, use your knowledge of names to infer the likely gender
   - For categories, use your knowledge to classify entities (e.g., `mammal(dog)`, `country(france)`)
   - Generate FACTS based on what you know about the world

2. **Derivable relationships:** Generate RULES when the predicate can be logically derived from existing KB predicates.

3. **Transitive/recursive predicates:** Generate RULES with base case and recursive case.

4. **NEVER generate facts that already exist in the KB.**

5. **ONLY output definitions for '{functor}' - do not define other predicates.**

6. **Common sense:** If a query seems false (e.g., `male(mary)`), generate the correct fact instead (e.g., `female(mary)`) or output nothing.

---

## Output Format
Think step-by-step, then provide your Prolog code in a ```prolog block.

```prolog
% Your facts and/or rules here
```

Now, analyze the query `{query}` and generate the appropriate definition for '{functor}':""",
                variables=["query", "context", "functor", "examples"]
            ),
            PromptTemplate(
                id="define_step_by_step",
                category=PromptCategory.DEFINITION,
                template="""Let's define '{functor}' step by step.

1. The query is: {query}
2. Existing knowledge: {context}
3. This predicate likely means: {hint}

Provide the definition as S-expression: (rule (head args...) ((body1 args...) (body2 args...)))""",
                variables=["query", "context", "functor", "hint"]
            )
        ])
        
        # ABSTRACTION templates
        self.templates[PromptCategory.ABSTRACTION].extend([
            PromptTemplate(
                id="find_abstraction",
                category=PromptCategory.ABSTRACTION,
                template="""What higher-level concept explains these patterns?
{patterns}

Create an abstract rule that captures the essence.
Output: (rule (abstract_predicate vars...) ((condition1 args...) (condition2 args...)))""",
                variables=["patterns"]
            )
        ])
        
        # EXAMPLE_GENERATION templates
        self.templates[PromptCategory.EXAMPLE_GENERATION].extend([
            PromptTemplate(
                id="generate_examples",
                category=PromptCategory.EXAMPLE_GENERATION,
                template="""Given this rule: {rule}

Generate {num_examples} example facts that would match this rule.
Output S-expressions, one per line:
(fact1 args...)
(fact2 args...)""",
                variables=["rule", "num_examples"]
            )
        ])
        
        # Initialize performance tracking (selector needs library reference, so skip for now)
        self.performance_data = {}
    
    def get_best_prompt(self, context: QueryContext) -> Tuple[str, str]:
        """
        Get the best prompt for the given context
        
        Returns:
            Tuple of (prompt_text, template_name)
        """
        # For now, use DEFINITION templates for queries
        # Later this can be more sophisticated based on context
        templates = self.templates[PromptCategory.DEFINITION]
        
        if not templates:
            # Fallback prompt
            prompt = f"""Query: {context.term}
Knowledge base has {len(context.kb_facts)} facts and {len(context.kb_rules)} rules.

Please generate relevant facts and rules for this query.
Output S-expressions:
(fact args...)
(rule (head args...) ((body args...)))"""
            return prompt, "fallback"
        
        # Choose template based on context
        if context.is_empty_kb:
            template = templates[0]  # Simple definition
        elif context.has_examples and len(context.kb_facts) > 3:
            # Use template with examples if we have them
            template = next((t for t in templates if "examples" in t.id), templates[0])
        else:
            template = templates[0]
        
        # Build prompt from template
        variables = {}
        if "query" in template.variables:
            # Format query as JSON
            variables["query"] = json.dumps(context.term)
        if "context" in template.variables:
            # Build context string with markdown headers using JSON format
            ctx_parts = []
            if context.kb_facts:
                # One fact per line in JSON format
                ctx_parts.append("### Facts")
                for fact in context.kb_facts[:5]:
                    ctx_parts.append(json.dumps(fact))
            if context.kb_rules:
                # Rules in JSON format: ["rule", head, body]
                ctx_parts.append("\n### Rules")
                for h, b in context.kb_rules[:3]:
                    rule_json = ["rule", h, b]
                    ctx_parts.append(json.dumps(rule_json))
            variables["context"] = "\n".join(ctx_parts) if ctx_parts else "Empty knowledge base"
        if "functor" in template.variables:
            # Extract functor from JSON term: ["functor", "arg1", ...]
            if isinstance(context.term, list) and len(context.term) > 0:
                functor = context.term[0]
            else:
                functor = "unknown"
            variables["functor"] = functor
        if "arity" in template.variables:
            # Count arguments in JSON term: ["functor", "arg1", "arg2"]
            if isinstance(context.term, list) and len(context.term) > 0:
                arity = len(context.term) - 1  # Minus the functor
            else:
                arity = 0
            variables["arity"] = arity
        if "examples" in template.variables:
            # Sample examples from the database
            num_examples = 5  # Default, can be made configurable
            temperature = 1.0  # Default temperature for softmax sampling
            functor = variables.get("functor", "unknown")

            # Build KB context string for retrieval
            kb_context = ""
            if context.kb_facts:
                kb_context = " ".join(str(f) for f in context.kb_facts[:10])

            # Use retriever if available, otherwise random sampling
            if self.example_retriever:
                sampled = self.example_retriever.retrieve(
                    functor, num_examples, temperature=temperature, kb_context=kb_context
                )
            else:
                sampled = sample_examples(num_examples)

            # Format examples as markdown (supports both old and new format)
            examples_text = []
            for i, ex in enumerate(sampled, 1):
                examples_text.append(f"**Example {i}**")
                # New format: query, output, kb_sample
                if "query" in ex and "output" in ex:
                    if ex.get("kb_sample"):
                        examples_text.append(f"KB: `{ex['kb_sample']}`")
                    examples_text.append(f"Query: `{ex['query']}`")
                    examples_text.append(f"```prolog\n{ex['output']}\n```")
                # Old format: prolog, json
                elif "prolog" in ex:
                    examples_text.append(f"```prolog\n{ex['prolog']}\n```")
                examples_text.append("")  # Blank line
            variables["examples"] = "\n".join(examples_text)

        prompt = template.render(**variables)
        return prompt, template.id
    
    def record_performance(self, template_name: str, success: bool, response_quality: float):
        """Record performance data for a template"""
        if template_name not in self.performance_data:
            self.performance_data[template_name] = {
                'successes': 0,
                'failures': 0,
                'total_quality': 0.0,
                'count': 0
            }
        
        data = self.performance_data[template_name]
        if success:
            data['successes'] += 1
        else:
            data['failures'] += 1
        data['total_quality'] += response_quality
        data['count'] += 1
    
    def select_template(self, context: QueryContext) -> PromptTemplate:
        """
        Select the best template for the given context
        
        Args:
            context: Query context
            
        Returns:
            Selected prompt template
        """
        # Determine category based on context
        category = self._determine_category(context)
        
        # Get templates for category
        templates = self.templates.get(category, [])
        if not templates:
            # Fall back to definition templates
            templates = self.templates.get(PromptCategory.DEFINITION, [])
        
        if not templates:
            # Create a default template
            return PromptTemplate(
                id="default",
                category=PromptCategory.DEFINITION,
                template="Query: {query}\nContext: {context}\nGenerate facts or rules in S-expression format.",
                variables=["query", "context"]
            )
        
        # For now, return the first template (could be smarter)
        return templates[0]
    
    def _determine_category(self, context: QueryContext) -> PromptCategory:
        """Determine the appropriate category based on context"""
        term = context.term
        
        # Check for specific patterns
        if "optimize" in term.lower() or "compress" in term.lower():
            return PromptCategory.COMPRESSION
        elif "abstract" in term.lower():
            return PromptCategory.ABSTRACTION
        elif "generalize" in term.lower():
            return PromptCategory.GENERALIZATION
        elif context.kb_facts and len(context.kb_facts) > 10:
            return PromptCategory.CONSOLIDATION
        else:
            return PromptCategory.DEFINITION
    
    def format_prompt(self, context: QueryContext, template_name: Optional[str] = None) -> str:
        """
        Format a prompt using the specified template or select the best one
        
        Args:
            context: Query context with term and KB information
            template_name: Optional specific template to use
        
        Returns:
            Formatted prompt string
        """
        if template_name:
            # Get specific template by name
            for templates in self.templates.values():
                for template in templates:
                    if template.id == template_name:
                        return self._format_template(template, context)
        
        # Select best template based on context
        template = self.select_template(context)
        return self._format_template(template, context)
    
    def _format_template(self, template: PromptTemplate, context: QueryContext) -> str:
        """Format a template with context variables"""
        # Build variable dictionary
        variables = {
            'query': context.term,
            'term': context.term,
            'context': self._format_context(context),
            'kb_facts': '\n'.join(context.kb_facts[:10]),
            'kb_rules': self._format_rules(context.kb_rules[:5]),
            'existing_functors': ', '.join(context.existing_functors[:20])
        }
        
        # Extract functor if it's a compound term
        if '(' in context.term:
            functor = context.term.split('(')[0]
            variables['functor'] = functor
            variables['arity'] = context.term.count(',') + 1
        
        # Format template
        prompt = template.template
        for var, value in variables.items():
            prompt = prompt.replace(f'{{{var}}}', str(value))
        
        return prompt
    
    def _format_context(self, context: QueryContext) -> str:
        """Format context information"""
        parts = []
        if context.kb_facts:
            parts.append(f"Known facts: {', '.join(context.kb_facts[:5])}")
        if context.kb_rules:
            parts.append(f"Known rules: {len(context.kb_rules)} rules")
        return '\n'.join(parts) if parts else "Empty knowledge base"
    
    def _format_rules(self, rules: List[Tuple[str, List[str]]]) -> str:
        """Format rules for display"""
        formatted = []
        for head, body in rules:
            body_str = ', '.join(body)
            formatted.append(f"{head} :- {body_str}")
        return '\n'.join(formatted)
    
    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics for all templates"""
        stats = {}
        for name, data in self.performance_data.items():
            if data['count'] > 0:
                stats[name] = {
                    'success_rate': data['successes'] / data['count'],
                    'avg_quality': data['total_quality'] / data['count'],
                    'count': data['count']
                }
        return stats


class AdaptivePromptSelector:
    """
    Selects and adapts prompts based on:
    - Query type
    - Model being used
    - Historical performance
    - Current context
    """
    
    def __init__(self, 
                 library: PromptTemplateLibrary,
                 model_params: Dict[str, ModelParameters]):
        self.library = library
        self.model_params = model_params
        self.selection_history = []
    
    def select_template(self,
                        query: str,
                        category: PromptCategory,
                        model: str,
                        context_size: int = 0) -> PromptTemplate:
        """
        Select the best template for the given context.
        
        Args:
            query: The query being processed
            category: Category of reasoning needed
            model: Model name
            context_size: Current context size in tokens (approximate)
        """
        templates = self.library.templates[category]
        if not templates:
            raise ValueError(f"No templates for category {category}")
        
        model_param = self.model_params.get(model)
        if not model_param:
            # Use first template if model unknown
            return templates[0]
        
        # Score each template
        scores = []
        for template in templates:
            score = self._score_template(template, model_param, context_size)
            scores.append((template, score))
        
        # Use weighted random selection (exploration vs exploitation)
        if random.random() < 0.1:  # 10% exploration
            return random.choice(templates)
        else:
            # Select based on scores
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[0][0]
    
    def _score_template(self, 
                       template: PromptTemplate,
                       model_param: ModelParameters,
                       context_size: int) -> float:
        """Score a template for the given context"""
        score = 0.0
        
        # Base success rate
        if model_param.model_name in template.model_success_rates:
            score += template.model_success_rates[model_param.model_name] * 10
        else:
            score += template.success_rate * 5
        
        # Penalize if context too large
        if context_size > model_param.max_context_tokens * 0.8:
            score -= 5
        
        # Bonus for frequently successful templates
        if template.use_count > 10 and template.success_rate > 0.8:
            score += 3
        
        # Category-specific model performance
        category_name = template.category.value
        if category_name in model_param.success_rates:
            score += model_param.success_rates[category_name] * 2
        
        return score
    
    def record_outcome(self,
                       template: PromptTemplate,
                       model: str,
                       success: bool,
                       response_quality: float = 0.5):
        """
        Record the outcome of using a template.
        
        Args:
            template: The template that was used
            model: Model name
            success: Whether the query succeeded
            response_quality: Quality score (0-1)
        """
        template.use_count += 1
        if success:
            template.success_count += 1
        
        # Update running average of quality
        alpha = 0.1  # Learning rate
        template.avg_response_quality = (
            (1 - alpha) * template.avg_response_quality + 
            alpha * response_quality
        )
        
        # Update model-specific success rate
        if model not in template.model_success_rates:
            template.model_success_rates[model] = 0.0
        
        old_rate = template.model_success_rates[model]
        template.model_success_rates[model] = (
            (1 - alpha) * old_rate + alpha * (1.0 if success else 0.0)
        )
        
        # Update model parameters
        if model in self.model_params:
            model_param = self.model_params[model]
            category = template.category.value
            
            if category not in model_param.success_rates:
                model_param.success_rates[category] = 0.5
            
            old_rate = model_param.success_rates[category]
            model_param.success_rates[category] = (
                (1 - alpha) * old_rate + alpha * (1.0 if success else 0.0)
            )


class PromptBuilder:
    """
    Builds complete prompts with examples and context.
    Handles model-specific formatting.
    """
    
    def __init__(self,
                 template_selector: AdaptivePromptSelector,
                 example_retriever=None):  # RAG system
        self.selector = template_selector
        self.example_retriever = example_retriever
    
    def build_prompt(self,
                    query: str,
                    category: PromptCategory,
                    model: str,
                    context: Dict[str, Any],
                    include_examples: bool = True) -> Tuple[str, PromptTemplate]:
        """
        Build a complete prompt for the given query.
        
        Returns:
            Tuple of (prompt_text, template_used)
        """
        # Get model parameters
        model_param = self.selector.model_params.get(
            model, 
            ModelParameters(model_name=model)
        )
        
        # Select template
        template = self.selector.select_template(
            query, category, model, 
            context_size=len(str(context))  # Rough estimate
        )
        
        # Prepare variables
        variables = {}
        
        # Add context variables
        if "query" in template.variables:
            variables["query"] = query
        if "context" in template.variables:
            variables["context"] = self._format_context(context, model_param)
        if "functor" in template.variables:
            variables["functor"] = self._extract_functor(query)
        if "arity" in template.variables:
            variables["arity"] = self._extract_arity(query)
        
        # Add examples if needed
        if include_examples and "examples" in template.variables:
            examples = self._select_examples(
                query, model_param, category
            )
            variables["examples"] = self._format_examples(examples, model_param)
        
        # Fill in any missing variables
        for var in template.variables:
            if var not in variables:
                variables[var] = f"[{var}]"  # Placeholder
        
        # Build the prompt
        prompt_text = template.template.format(**variables)
        
        # Add model-specific formatting
        if model_param.needs_explicit_format:
            prompt_text = self._add_format_instructions(prompt_text, model_param)
        
        return prompt_text, template
    
    def _format_context(self, context: Dict[str, Any], model_param: ModelParameters) -> str:
        """Format context for the prompt"""
        if model_param.prefers_json:
            return json.dumps(context, indent=2)
        else:
            # Human-readable format
            lines = []
            if "facts" in context:
                lines.append(f"Facts: {context['facts']}")
            if "rules" in context:
                lines.append(f"Rules: {context['rules']}")
            return "\n".join(lines)
    
    def _select_examples(self, 
                        query: str,
                        model_param: ModelParameters,
                        category: PromptCategory) -> List[Dict]:
        """Select examples using the configured strategy"""
        if not self.example_retriever:
            return []
        
        num_examples = model_param.optimal_examples
        
        if model_param.example_selection_strategy == "similarity":
            # Get most similar examples
            examples = self.example_retriever.retrieve(
                query, k=num_examples
            )
        elif model_param.example_selection_strategy == "diversity":
            # Get diverse examples (TODO: implement clustering)
            examples = self.example_retriever.retrieve(
                query, k=num_examples * 2
            )
            # Sample for diversity
            examples = random.sample(examples, min(num_examples, len(examples)))
        else:  # mixed
            # Half similar, half random
            similar = self.example_retriever.retrieve(
                query, k=num_examples // 2
            )
            random_examples = self.example_retriever.retrieve(
                query, k=num_examples
            )
            examples = similar + random.sample(
                random_examples, 
                min(num_examples - len(similar), len(random_examples))
            )
        
        return examples
    
    def _format_examples(self, examples: List[Dict], model_param: ModelParameters) -> str:
        """Format examples for inclusion in prompt"""
        if not examples:
            return "No examples available"
        
        formatted = []
        for i, ex in enumerate(examples[:model_param.max_examples], 1):
            formatted.append(f"Example {i}: {json.dumps(ex)}")
        
        return "\n".join(formatted)
    
    def _extract_functor(self, query: str) -> str:
        """Extract the main functor from a query"""
        # Simple extraction - can be improved
        if "(" in query:
            return query.split("(")[1].split()[0]
        return "unknown"
    
    def _extract_arity(self, query: str) -> int:
        """Extract arity (number of arguments) from query"""
        # Simple counting - can be improved
        if "(" in query and ")" in query:
            args = query[query.index("(")+1:query.index(")")].split()
            return len(args) - 1  # Subtract functor
        return 0
    
    def _add_format_instructions(self, prompt: str, model_param: ModelParameters) -> str:
        """Add model-specific format instructions"""
        if model_param.prefers_json:
            prompt += "\n\nIMPORTANT: Return only valid JSON, no other text."
        return prompt


class PromptLearningSystem:
    """
    System that learns which prompts work best over time.
    Implements meta-learning for prompt optimization.
    """
    
    def __init__(self, 
                 library: PromptTemplateLibrary,
                 save_path: Optional[Path] = None):
        self.library = library
        self.save_path = save_path or Path.home() / ".dreamlog" / "prompt_learning.json"
        self.learning_history = []
        
        # Load existing learning if available
        if self.save_path.exists():
            self.load()
    
    def analyze_patterns(self) -> Dict[str, Any]:
        """Analyze what makes prompts successful"""
        patterns = {
            "best_templates_by_category": {},
            "optimal_example_counts": {},
            "successful_patterns": [],
            "failure_patterns": []
        }
        
        # Find best templates per category
        for category in PromptCategory:
            templates = self.library.templates[category]
            if templates:
                best = max(templates, key=lambda t: t.success_rate)
                patterns["best_templates_by_category"][category.value] = {
                    "id": best.id,
                    "success_rate": best.success_rate,
                    "use_count": best.use_count
                }
        
        # Analyze successful patterns
        for entry in self.learning_history:
            if entry.get("success"):
                patterns["successful_patterns"].append({
                    "category": entry["category"],
                    "model": entry["model"],
                    "template_id": entry["template_id"],
                    "context_size": entry.get("context_size", 0)
                })
        
        return patterns
    
    def generate_new_template(self, 
                             category: PromptCategory,
                             based_on: List[PromptTemplate]) -> PromptTemplate:
        """
        Generate a new template based on successful ones.
        This is where meta-learning happens.
        """
        # Simple crossover for now - can be made more sophisticated
        if len(based_on) < 2:
            return None
        
        parent1, parent2 = random.sample(based_on, 2)
        
        # Combine elements from both templates
        new_template = PromptTemplate(
            id=f"generated_{category.value}_{len(self.library.templates[category])}",
            category=category,
            template=self._crossover_templates(parent1.template, parent2.template),
            variables=list(set(parent1.variables + parent2.variables))
        )
        
        return new_template
    
    def _crossover_templates(self, template1: str, template2: str) -> str:
        """Combine two templates to create a new one"""
        # Simple approach: take first half of one, second half of other
        lines1 = template1.split("\n")
        lines2 = template2.split("\n")
        
        mid1 = len(lines1) // 2
        mid2 = len(lines2) // 2
        
        new_lines = lines1[:mid1] + lines2[mid2:]
        return "\n".join(new_lines)
    
    def save(self):
        """Save learning history and template performance"""
        data = {
            "learning_history": self.learning_history,
            "template_performance": {}
        }
        
        for category in PromptCategory:
            data["template_performance"][category.value] = [
                {
                    "id": t.id,
                    "use_count": t.use_count,
                    "success_count": t.success_count,
                    "avg_quality": t.avg_response_quality,
                    "model_success_rates": t.model_success_rates
                }
                for t in self.library.templates[category]
            ]
        
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.save_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self):
        """Load learning history and update template performance"""
        try:
            with open(self.save_path, 'r') as f:
                data = json.load(f)
            
            self.learning_history = data.get("learning_history", [])
            
            # Update template performance
            for category_name, templates_data in data.get("template_performance", {}).items():
                category = PromptCategory(category_name)
                for template_data in templates_data:
                    # Find matching template
                    for template in self.library.templates[category]:
                        if template.id == template_data["id"]:
                            template.use_count = template_data["use_count"]
                            template.success_count = template_data["success_count"]
                            template.avg_response_quality = template_data["avg_quality"]
                            template.model_success_rates = template_data["model_success_rates"]
                            break
        except Exception as e:
            print(f"Error loading prompt learning data: {e}")