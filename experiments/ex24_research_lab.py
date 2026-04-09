#!/usr/bin/env python3
"""
EX24: Organic research lab knowledge graph.

Simulates a PI (Dr. Chen) building institutional memory for their research
group over ~25 "sessions" spanning two years. Facts accumulate organically
across seven domains:

  1. People (students, postdocs, faculty, staff, alumni, collaborators)
  2. Publications (papers, authors, venues, topics, methods, citations)
  3. Teaching (courses, prerequisites, semesters, enrollments, TAs)
  4. Grants (funding agencies, PIs, topics, budgets, periods)
  5. Infrastructure (servers, GPUs, datasets, software, access)
  6. Collaborations (external co-authors, joint projects, shared students)
  7. Service (reviews, committees, talks, organizing)

The KB grows messily: some sessions add 5 facts, others add 30. Corrections
happen mid-stream. Queries evolve from simple lookups to multi-hop reasoning.
Patterns only become discoverable after sufficient cross-domain data.

Designed to stress-test every dream operation:
  - Op A: subsumption from user-provided general rules
  - Op B: facts made redundant by discovered rules
  - Op C: generalization with exceptions (most students funded, one isn't)
  - Op D: predicate invention from structurally identical rule sets
  - Op E: body pattern extraction from shared sub-goal sequences
  - Op G: cross-domain rules the symbolic ops can't find
  - Op H: frequently derived intermediate results cached as lemma

Usage:
    python experiments/ex24_research_lab.py
    python experiments/ex24_research_lab.py --dream-every 5
    python experiments/ex24_research_lab.py --store /tmp/lab.json
"""

import sys
import time
import json
from pathlib import Path

sys.path.insert(0, ".")

from integrations.mcp.knowledge_store import KnowledgeStore
from dreamlog.terms import Compound


# ── Session data ─────────────────────────────────────────────────────

SESSIONS = [
    # ================================================================
    # PHASE 1: Foundation (sessions 1-5) — basic people and structure
    # ================================================================
    {
        "name": "S01: PI and initial group",
        "description": "Dr. Chen starts tracking their lab",
        "assertions": [
            # PI
            "(person chen)", "(role chen pi)",
            "(department chen cs)", "(institution chen stateu)",
            "(expertise chen ml)", "(expertise chen nlp)",
            # First students
            "(person amir)", "(role amir phd_student)",
            "(advisor amir chen)", "(year_started amir 2022)",
            "(expertise amir nlp)", "(expertise amir transformers)",
            "(person bea)", "(role bea phd_student)",
            "(advisor bea chen)", "(year_started bea 2021)",
            "(expertise bea ml)", "(expertise bea cv)",
            # Postdoc
            "(person carlos)", "(role carlos postdoc)",
            "(advisor carlos chen)", "(year_started carlos 2023)",
            "(expertise carlos rl)", "(expertise carlos ml)",
        ],
        "queries": [
            "(advisor X chen)",
            "(expertise chen X)",
        ],
    },
    {
        "name": "S02: More group members",
        "assertions": [
            "(person diana)", "(role diana masters_student)",
            "(advisor diana chen)", "(year_started diana 2023)",
            "(expertise diana nlp)",
            "(person elena)", "(role elena phd_student)",
            "(advisor elena chen)", "(year_started elena 2023)",
            "(expertise elena ml)", "(expertise elena optimization)",
            # Lab manager
            "(person frank)", "(role frank lab_manager)",
            "(department frank cs)",
            # Undergrad researcher
            "(person gita)", "(role gita undergrad_ra)",
            "(mentor gita amir)", "(year_started gita 2023)",
        ],
        "queries": [
            "(role X phd_student)",
            "(year_started X 2023)",
        ],
    },
    {
        "name": "S03: Courses and teaching",
        "assertions": [
            "(course cs571 ml_intro)", "(course cs672 deep_learning)",
            "(course cs573 nlp_fundamentals)", "(course cs780 adv_topics_ai)",
            "(prerequisite cs672 cs571)",
            "(prerequisite cs780 cs672)",
            "(prerequisite cs573 cs571)",
            "(teaches chen cs672 fall2023)", "(teaches chen cs780 spring2024)",
            "(ta amir cs573 fall2023)", "(ta bea cs672 fall2023)",
            "(enrolled diana cs672 fall2023)",
            "(enrolled elena cs780 spring2024)",
            "(enrolled gita cs571 fall2023)",
        ],
        "queries": [
            "(teaches chen X fall2023)",
            "(prerequisite cs780 X)",
        ],
    },
    {
        "name": "S04: First publications",
        "assertions": [
            "(paper p001 transformer_pruning)",
            "(author p001 chen)", "(author p001 amir)",
            "(venue p001 acl2023)", "(year p001 2023)",
            "(topic p001 nlp)", "(topic p001 efficiency)",
            "(method p001 pruning)", "(method p001 transformers)",
            "(paper p002 visual_rl)",
            "(author p002 chen)", "(author p002 bea)", "(author p002 carlos)",
            "(venue p002 neurips2023)", "(year p002 2023)",
            "(topic p002 rl)", "(topic p002 cv)",
            "(method p002 reinforcement_learning)",
        ],
        "queries": [
            "(author X chen)",
            "(topic p001 X)",
        ],
    },
    {
        "name": "S05: Lab infrastructure",
        "assertions": [
            "(server gpu_cluster_1)", "(gpu_count gpu_cluster_1 8)",
            "(gpu_type gpu_cluster_1 a100)",
            "(server gpu_cluster_2)", "(gpu_count gpu_cluster_2 4)",
            "(gpu_type gpu_cluster_2 v100)",
            "(server cpu_server)", "(cpu_cores cpu_server 128)",
            "(has_access amir gpu_cluster_1)",
            "(has_access bea gpu_cluster_1)",
            "(has_access carlos gpu_cluster_2)",
            "(has_access elena gpu_cluster_2)",
            "(has_access frank gpu_cluster_1)",
            "(has_access frank gpu_cluster_2)",
            "(has_access frank cpu_server)",
            "(dataset imagenet)", "(dataset squad)", "(dataset glue)",
            "(uses_dataset amir glue)", "(uses_dataset bea imagenet)",
        ],
        "queries": [
            "(has_access X gpu_cluster_1)",
            "(server X)",
        ],
    },

    # ================================================================
    # PHASE 2: Growth (sessions 6-10) — grants, collaborations, depth
    # ================================================================
    {
        "name": "S06: Grants and funding",
        "assertions": [
            "(grant nsf_2301 nsf)", "(grant_pi nsf_2301 chen)",
            "(grant_topic nsf_2301 nlp)", "(grant_topic nsf_2301 efficiency)",
            "(grant_amount nsf_2301 500000)", "(grant_period nsf_2301 2023 2026)",
            "(funded_by amir nsf_2301)", "(funded_by diana nsf_2301)",
            "(grant nih_r01 nih)", "(grant_pi nih_r01 chen)",
            "(grant_topic nih_r01 ml)", "(grant_topic nih_r01 medical_imaging)",
            "(grant_amount nih_r01 800000)", "(grant_period nih_r01 2022 2025)",
            "(funded_by bea nih_r01)",
            # Elena is unfunded — exception to "all PhD students funded"
            "(self_funded elena)",
        ],
        "queries": [
            "(funded_by X nsf_2301)",
            "(grant_pi X chen)",
        ],
    },
    {
        "name": "S07: External collaborators",
        "assertions": [
            "(person prof_kim)", "(role prof_kim faculty)",
            "(institution prof_kim mit)", "(expertise prof_kim nlp)",
            "(expertise prof_kim linguistics)",
            "(person prof_lee)", "(role prof_lee faculty)",
            "(institution prof_lee stanford)", "(expertise prof_lee rl)",
            "(expertise prof_lee robotics)",
            "(collaborator chen prof_kim)",
            "(collaborator chen prof_lee)",
            # Joint paper with Kim
            "(paper p003 crosslingual_transfer)",
            "(author p003 chen)", "(author p003 amir)", "(author p003 prof_kim)",
            "(venue p003 emnlp2023)", "(year p003 2023)",
            "(topic p003 nlp)", "(topic p003 transfer_learning)",
            "(method p003 transformers)",
        ],
        "queries": [
            "(collaborator chen X)",
            "(author p003 X)",
            "(institution X mit)",
        ],
    },
    {
        "name": "S08: More papers and citations",
        "assertions": [
            "(paper p004 optimization_landscape)",
            "(author p004 chen)", "(author p004 elena)",
            "(venue p004 icml2024)", "(year p004 2024)",
            "(topic p004 optimization)", "(topic p004 ml)",
            "(method p004 gradient_analysis)",
            # Citations between papers
            "(cites p004 p001)", "(cites p004 p002)",
            "(cites p003 p001)",
            # External citations
            "(paper p_ext1 attention_survey)", "(year p_ext1 2022)",
            "(cites p001 p_ext1)", "(cites p003 p_ext1)",
            # Workshop paper
            "(paper p005 efficient_finetuning)",
            "(author p005 amir)", "(author p005 diana)",
            "(venue p005 emnlp_workshop_2023)", "(year p005 2023)",
            "(topic p005 nlp)", "(topic p005 efficiency)",
            "(method p005 pruning)", "(method p005 finetuning)",
        ],
        "queries": [
            "(cites p004 X)",
            "(venue X icml2024)",
            "(author p005 X)",
        ],
    },
    {
        "name": "S09: Service and reviews",
        "assertions": [
            "(reviews_for chen acl2024)",
            "(reviews_for chen neurips2024)",
            "(reviews_for amir emnlp2024)",
            "(committee chen hiring_2024)",
            "(committee chen curriculum_2024)",
            "(committee bea grad_council_2024)",
            "(talk chen invited stanford march2024)",
            "(talk chen keynote acl_workshop april2024)",
            "(talk amir contributed emnlp2023 oct2023)",
            "(organizing chen workshop_chair emnlp_efficient2024)",
        ],
        "queries": [
            "(reviews_for chen X)",
            "(talk chen X Y Z)",
        ],
    },
    {
        "name": "S10: Corrections and updates",
        "description": "Mid-stream corrections — Carlos leaves, new people join",
        "assertions": [
            # Carlos's postdoc ends
            "(alumni carlos 2024)", "(current_position carlos industry)",
            "(affiliation carlos google_brain)",
            # New postdoc arrives
            "(person hiro)", "(role hiro postdoc)",
            "(advisor hiro chen)", "(year_started hiro 2024)",
            "(expertise hiro ml)", "(expertise hiro generative_models)",
            # Gita graduates, becomes masters student
            "(role gita masters_student)",  # overwrites undergrad_ra conceptually
            "(advisor gita chen)",  # now directly advised
            # More infrastructure
            "(server gpu_cluster_3)", "(gpu_count gpu_cluster_3 8)",
            "(gpu_type gpu_cluster_3 h100)",
            "(has_access hiro gpu_cluster_3)",
            "(has_access amir gpu_cluster_3)",
            "(has_access elena gpu_cluster_3)",
        ],
        "queries": [
            "(alumni X 2024)",
            "(has_access X gpu_cluster_3)",
            "(advisor X chen)",
        ],
    },

    # ================================================================
    # PHASE 3: Maturity (sessions 11-17) — deep knowledge, patterns
    # ================================================================
    {
        "name": "S11: Student milestones and progress",
        "assertions": [
            "(passed_quals amir 2023)", "(passed_quals bea 2023)",
            "(passed_quals elena 2024)",
            "(dissertation_proposal amir 2024)",
            "(dissertation_topic amir efficient_nlp)",
            "(first_author_paper amir p001)",
            "(first_author_paper amir p005)",
            "(first_author_paper bea p002)",
            "(first_author_paper elena p004)",
            # Amir is close to graduating
            "(expected_graduation amir 2025)",
            "(expected_graduation bea 2026)",
            "(expected_graduation elena 2027)",
        ],
        "queries": [
            "(passed_quals X 2023)",
            "(expected_graduation X 2025)",
            "(first_author_paper amir X)",
        ],
    },
    {
        "name": "S12: Software and tools",
        "assertions": [
            "(software prunetransformer)", "(author_sw prunetransformer amir)",
            "(language_sw prunetransformer python)",
            "(repo_sw prunetransformer github)",
            "(open_source prunetransformer)",
            "(software vrl_benchmark)", "(author_sw vrl_benchmark bea)",
            "(author_sw vrl_benchmark carlos)",
            "(language_sw vrl_benchmark python)",
            "(repo_sw vrl_benchmark github)",
            "(open_source vrl_benchmark)",
            "(software optland_viz)", "(author_sw optland_viz elena)",
            "(language_sw optland_viz python)",
            "(repo_sw optland_viz internal)",
            # Software-paper links
            "(implements prunetransformer p001)",
            "(implements vrl_benchmark p002)",
            "(implements optland_viz p004)",
        ],
        "queries": [
            "(open_source X)",
            "(implements X p001)",
            "(author_sw X amir)",
        ],
    },
    {
        "name": "S13: New grant and collaboration expansion",
        "assertions": [
            "(grant darpa_2401 darpa)", "(grant_pi darpa_2401 chen)",
            "(grant_copi darpa_2401 prof_kim)",
            "(grant_topic darpa_2401 nlp)",
            "(grant_topic darpa_2401 efficiency)",
            "(grant_topic darpa_2401 deployment)",
            "(grant_amount darpa_2401 1200000)",
            "(grant_period darpa_2401 2024 2027)",
            "(funded_by amir darpa_2401)",  # Amir now on DARPA grant
            "(funded_by hiro darpa_2401)",
            # New external collaborator
            "(person dr_patel)", "(role dr_patel faculty)",
            "(institution dr_patel cmu)", "(expertise dr_patel systems)",
            "(expertise dr_patel efficiency)",
            "(collaborator chen dr_patel)",
            # Joint work
            "(paper p006 efficient_inference)",
            "(author p006 chen)", "(author p006 hiro)", "(author p006 dr_patel)",
            "(venue p006 mlsys2024)", "(year p006 2024)",
            "(topic p006 efficiency)", "(topic p006 systems)",
            "(method p006 quantization)",
        ],
        "queries": [
            "(grant_copi darpa_2401 X)",
            "(funded_by X darpa_2401)",
            "(collaborator chen X)",
        ],
    },
    {
        "name": "S14: Datasets and experiments tracking",
        "assertions": [
            "(dataset clinical_notes)", "(dataset arxiv_abstracts)",
            "(dataset code_search_net)",
            "(dataset_license imagenet research_only)",
            "(dataset_license squad open)", "(dataset_license glue open)",
            "(dataset_license clinical_notes restricted)",
            "(dataset_license arxiv_abstracts open)",
            "(dataset_license code_search_net open)",
            "(uses_dataset amir arxiv_abstracts)",
            "(uses_dataset diana glue)",
            "(uses_dataset hiro code_search_net)",
            "(uses_dataset elena imagenet)",
            # Experiment tracking
            "(experiment exp001 amir pruning_ablation gpu_cluster_1)",
            "(experiment exp002 bea visual_pretraining gpu_cluster_1)",
            "(experiment exp003 elena landscape_sweep gpu_cluster_2)",
            "(experiment exp004 hiro gen_model_train gpu_cluster_3)",
        ],
        "queries": [
            "(dataset_license X open)",
            "(experiment X amir Y Z)",
            "(uses_dataset hiro X)",
        ],
    },
    {
        "name": "S15: Teaching load expansion",
        "assertions": [
            "(teaches chen cs571 fall2024)",  # now teaches intro too
            "(teaches chen cs780 spring2025)",
            "(ta diana cs571 fall2024)",
            "(ta elena cs672 fall2024)",
            "(enrolled gita cs672 fall2024)",
            "(enrolled gita cs573 fall2024)",
            # Guest lectures
            "(guest_lecture amir cs672 transformers fall2024)",
            "(guest_lecture hiro cs780 generative_models spring2025)",
            # Amir co-teaches a reading group
            "(leads_seminar amir reading_group_nlp fall2024)",
            "(attends_seminar bea reading_group_nlp fall2024)",
            "(attends_seminar diana reading_group_nlp fall2024)",
        ],
        "queries": [
            "(teaches chen X fall2024)",
            "(ta X Y fall2024)",
            "(guest_lecture X cs672 Y Z)",
        ],
    },
    {
        "name": "S16: More publications — productivity burst",
        "assertions": [
            # Amir's dissertation papers
            "(paper p007 pruning_theory)",
            "(author p007 chen)", "(author p007 amir)",
            "(venue p007 iclr2024)", "(year p007 2024)",
            "(topic p007 nlp)", "(topic p007 efficiency)", "(topic p007 theory)",
            "(method p007 pruning)", "(method p007 analysis)",
            "(first_author_paper amir p007)",
            # Diana's first paper
            "(paper p008 multilingual_efficiency)",
            "(author p008 chen)", "(author p008 diana)", "(author p008 prof_kim)",
            "(venue p008 naacl2024)", "(year p008 2024)",
            "(topic p008 nlp)", "(topic p008 efficiency)",
            "(topic p008 multilingual)",
            "(method p008 finetuning)", "(method p008 transformers)",
            "(first_author_paper diana p008)",
            # Hiro's generative paper
            "(paper p009 conditional_generation)",
            "(author p009 chen)", "(author p009 hiro)",
            "(venue p009 icml2024)", "(year p009 2024)",
            "(topic p009 ml)", "(topic p009 generative_models)",
            "(method p009 diffusion)",
            "(first_author_paper hiro p009)",
            # Citations
            "(cites p007 p001)", "(cites p007 p005)",
            "(cites p008 p001)", "(cites p008 p003)",
            "(cites p009 p002)",
        ],
        "queries": [
            "(first_author_paper X Y)",
            "(venue X iclr2024)",
            "(method X pruning)",
        ],
    },
    {
        "name": "S17: Explicit rules the user discovers",
        "description": "User notices patterns and asserts rules directly",
        "assertions": [
            # User realizes these relationships
            "(group_member X) :- (advisor X chen)",
            # Helper for existential check (NAF requires ground goals)
            "(is_alumni X) :- (alumni X Y)",
            "(active_member X) :- (group_member X), (not (is_alumni X))",
            # A student is senior if they've passed quals
            "(has_passed_quals X) :- (passed_quals X Y)",
            "(senior_student X) :- (role X phd_student), (has_passed_quals X)",
            # Can supervise undergrads if senior student
            "(can_mentor X) :- (senior_student X)",
            "(can_mentor X) :- (role X postdoc)",
        ],
        "queries": [
            "(group_member X)",
            "(active_member X)",
            "(senior_student X)",
            "(can_mentor X)",
        ],
    },

    # ================================================================
    # PHASE 4: Late stage (sessions 18-25) — complex patterns, alumni
    # ================================================================
    {
        "name": "S18: Conference travel and networking",
        "assertions": [
            "(attended_conf amir acl2023)", "(attended_conf chen acl2023)",
            "(attended_conf amir emnlp2023)", "(attended_conf chen emnlp2023)",
            "(attended_conf bea neurips2023)", "(attended_conf chen neurips2023)",
            "(attended_conf carlos neurips2023)",
            "(attended_conf elena icml2024)", "(attended_conf chen icml2024)",
            "(attended_conf hiro icml2024)",
            "(attended_conf amir iclr2024)", "(attended_conf chen iclr2024)",
            "(attended_conf diana naacl2024)", "(attended_conf chen naacl2024)",
            # Met at conferences — potential collaborators
            "(met_at chen prof_zhang neurips2023)",
            "(met_at chen prof_wang icml2024)",
            "(person prof_zhang)", "(institution prof_zhang berkeley)",
            "(expertise prof_zhang ml)", "(expertise prof_zhang fairness)",
            "(person prof_wang)", "(institution prof_wang eth)",
            "(expertise prof_wang optimization)", "(expertise prof_wang ml)",
        ],
        "queries": [
            "(attended_conf X icml2024)",
            "(met_at chen X Y)",
        ],
    },
    {
        "name": "S19: Bea's dissertation and defense",
        "assertions": [
            "(dissertation_proposal bea 2024)",
            "(dissertation_topic bea visual_reinforcement_learning)",
            "(paper p010 visual_rl_survey)",
            "(author p010 bea)", "(author p010 chen)",
            "(venue p010 jmlr2024)", "(year p010 2024)",
            "(topic p010 rl)", "(topic p010 cv)", "(topic p010 survey)",
            "(method p010 survey_methodology)",
            "(first_author_paper bea p010)",
            # Bea on job market
            "(job_market bea 2024)",
            "(reference_letter chen bea)",
            "(reference_letter prof_lee bea)",
        ],
        "queries": [
            "(dissertation_topic bea X)",
            "(reference_letter X bea)",
            "(first_author_paper bea X)",
        ],
    },
    {
        "name": "S20: Lab awards and recognition",
        "assertions": [
            "(award amir best_paper acl2023)",
            "(award bea outstanding_reviewer neurips2023)",
            "(award elena rising_star icml2024)",
            "(award chen fellow aaai 2024)",
            "(award hiro best_demo mlsys2024)",
            # h-index and citation tracking
            "(h_index chen 42)", "(total_citations chen 8500)",
            "(h_index amir 6)", "(total_citations amir 320)",
            "(h_index bea 5)", "(total_citations bea 280)",
        ],
        "queries": [
            "(award X best_paper Y)",
            "(h_index chen X)",
        ],
    },
    {
        "name": "S21: New students and lab growth",
        "assertions": [
            # New PhD students
            "(person jun)", "(role jun phd_student)",
            "(advisor jun chen)", "(year_started jun 2024)",
            "(expertise jun nlp)", "(expertise jun safety)",
            "(funded_by jun nsf_2301)",
            "(person kate)", "(role kate phd_student)",
            "(advisor kate chen)", "(year_started kate 2024)",
            "(expertise kate ml)", "(expertise kate medical_imaging)",
            "(funded_by kate nih_r01)",
            # New undergrad
            "(person leo)", "(role leo undergrad_ra)",
            "(mentor leo elena)", "(year_started leo 2024)",
            # Gita finishes masters
            "(alumni gita 2024)", "(current_position gita industry)",
            "(affiliation gita meta)",
        ],
        "queries": [
            "(advisor X chen)",
            "(funded_by X nsf_2301)",
            "(alumni X 2024)",
        ],
    },
    {
        "name": "S22: Cross-domain experiment tracking",
        "assertions": [
            "(experiment exp005 amir distill_benchmark gpu_cluster_3)",
            "(experiment exp006 diana multilingual_eval gpu_cluster_1)",
            "(experiment exp007 jun safety_probe gpu_cluster_2)",
            "(experiment exp008 kate medical_pretrain gpu_cluster_3)",
            "(experiment exp009 hiro diffusion_scale gpu_cluster_3)",
            # Link experiments to papers
            "(supports_paper exp001 p001)",
            "(supports_paper exp002 p002)",
            "(supports_paper exp003 p004)",
            "(supports_paper exp005 p007)",
            "(supports_paper exp006 p008)",
            # Experiment dependencies
            "(depends_on_exp exp005 exp001)",
            "(depends_on_exp exp006 exp001)",
        ],
        "queries": [
            "(experiment X amir Y Z)",
            "(supports_paper X p001)",
            "(depends_on_exp exp005 X)",
        ],
    },
    {
        "name": "S23: Publication pipeline",
        "assertions": [
            "(paper p011 safety_alignment)",
            "(author p011 chen)", "(author p011 jun)",
            "(venue p011 under_review)", "(year p011 2025)",
            "(topic p011 nlp)", "(topic p011 safety)",
            "(method p011 probing)",
            "(paper p012 medical_foundation)",
            "(author p012 chen)", "(author p012 kate)", "(author p012 bea)",
            "(venue p012 under_review)", "(year p012 2025)",
            "(topic p012 ml)", "(topic p012 medical_imaging)",
            "(method p012 foundation_models)",
            "(paper p013 efficient_diffusion)",
            "(author p013 hiro)", "(author p013 chen)", "(author p013 dr_patel)",
            "(venue p013 neurips2025)", "(year p013 2025)",
            "(topic p013 efficiency)", "(topic p013 generative_models)",
            "(method p013 diffusion)", "(method p013 quantization)",
            "(first_author_paper jun p011)",
            "(first_author_paper kate p012)",
            "(first_author_paper hiro p013)",
            "(cites p011 p001)", "(cites p012 p002)", "(cites p012 p010)",
            "(cites p013 p006)", "(cites p013 p009)",
        ],
        "queries": [
            "(venue X under_review)",
            "(first_author_paper jun X)",
            "(topic X safety)",
        ],
    },
    {
        "name": "S24: Amir graduates — knowledge transfer",
        "assertions": [
            "(alumni amir 2025)",
            "(current_position amir postdoc)",
            "(affiliation amir mit)",  # joins Prof Kim's group
            "(thesis amir efficient_transformers_for_nlp 2025)",
            "(thesis_committee amir chen)",
            "(thesis_committee amir prof_kim)",
            "(thesis_committee amir dr_patel)",
            # Amir's legacy — who takes over his work
            "(successor amir diana)",  # Diana continues pruning work
            "(transferred_access amir diana gpu_cluster_1)",
            "(transferred_access amir diana gpu_cluster_3)",
            "(maintains diana prunetransformer)",  # Diana maintains the software
        ],
        "queries": [
            "(alumni X 2025)",
            "(successor amir X)",
            "(thesis_committee amir X)",
        ],
    },
    {
        "name": "S25: Retrospective — lab health check",
        "description": "Final session: user queries across all domains",
        "assertions": [
            # A few final facts
            "(paper_count chen 13)",  # Total papers in the system
            "(student_count chen 6)",  # Current + alumni PhD students
            "(grant_count chen 3)",
            "(lab_size chen 8)",  # Current active members
            # User-provided transitivity rule for citations
            "(indirectly_cites X Z) :- (cites X Y), (cites Y Z)",
        ],
        "queries": [
            # Simple lookups
            "(role X phd_student)",
            "(active_member X)",
            # Multi-hop
            "(indirectly_cites p007 X)",
            # Cross-domain
            "(expertise X nlp)",
            "(funded_by X nsf_2301)",
            "(first_author_paper X Y)",
            # Completeness
            "(alumni X Y)",
        ],
    },
]


def run_simulation():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="anthropic")
    parser.add_argument("--api-key-env", default="MY_ANTHROPIC_API_KEY")
    parser.add_argument("--dream-every", type=int, default=5,
                        help="Dream every N sessions")
    parser.add_argument("--store", default=None,
                        help="Store path (default: temp)")
    args = parser.parse_args()

    import tempfile
    store_path = (Path(args.store) if args.store
                  else Path(tempfile.mktemp(suffix=".json")))

    store = KnowledgeStore(store_path, llm_budget_usd=5.0,
                           dream_threshold=999)

    # Configure LLM
    if args.provider == "anthropic":
        from dreamlog.llm_client import LLMClient
        store.llm_client = LLMClient(
            provider="anthropic", api_key_env=args.api_key_env,
            temperature=0.3, max_tokens=1200)

    total_assertions = 0
    for s in SESSIONS:
        total_assertions += len(s["assertions"])

    print(f"{'='*72}")
    print(f"  EX24: Organic Research Lab Knowledge Graph")
    print(f"  Store: {store_path}")
    print(f"  Sessions: {len(SESSIONS)}, Total assertions: {total_assertions}")
    print(f"  Dream every {args.dream_every} sessions")
    print(f"{'='*72}")

    timeline = []

    for i, session in enumerate(SESSIONS, 1):
        label = session["name"]
        desc = session.get("description", "")
        print(f"\n  --- {label} ---", flush=True)
        if desc:
            print(f"    ({desc})", flush=True)

        # Assert facts/rules
        for fact in session["assertions"]:
            store.assert_fact(fact)

        # Run queries and report
        for query in session["queries"]:
            result = store.query(query)
            if result["count"] > 0:
                bindings_str = ", ".join(
                    str(s) for s in result["solutions"][:3])
                extra = f" (+{result['count']-3} more)" if result["count"] > 3 else ""
                print(f"    Q: {query} → {result['count']} "
                      f"results{extra}", flush=True)
            else:
                print(f"    Q: {query} → 0 results", flush=True)

        snapshot = {
            "session": i,
            "name": label,
            "facts_added": len(session["assertions"]),
            "queries_run": len(session["queries"]),
            "total_clauses": len(store.kb),
            "facts": len(store.kb.facts),
            "rules": len(store.kb.rules),
            "dreams": store._dream_count,
        }

        # Dream periodically
        if i % args.dream_every == 0:
            print(f"    💤 Dreaming...", flush=True)
            t0 = time.perf_counter()
            result = store.dream()
            elapsed = time.perf_counter() - t0

            snapshot["dream"] = {
                "before": result["before"],
                "after": result["after"],
                "removed": result["removed"],
                "ratio": round(result["ratio"], 3),
                "operations": result["operations"],
                "new_rules": result["new_rules"],
                "time": round(elapsed, 1),
                "llm_used": result["llm_used"],
            }

            n_rules = len(result["new_rules"])
            if result["new_rules"]:
                print(f"    Discovered {n_rules} rules:", flush=True)
                for r in result["new_rules"][:5]:
                    print(f"      + {r}", flush=True)
                if n_rules > 5:
                    print(f"      ... and {n_rules - 5} more", flush=True)
            print(f"    {result['before']} → {result['after']} "
                  f"(ratio {result['ratio']:.3f}) in {elapsed:.1f}s",
                  flush=True)

            snapshot["total_clauses"] = len(store.kb)
            snapshot["facts"] = len(store.kb.facts)
            snapshot["rules"] = len(store.kb.rules)

        print(f"    KB: {snapshot['total_clauses']} clauses "
              f"({snapshot['facts']}F + {snapshot['rules']}R)", flush=True)

        timeline.append(snapshot)

    # ── Final dream ──────────────────────────────────────────────
    print(f"\n  --- Final dream ---", flush=True)
    t0 = time.perf_counter()
    final = store.dream()
    elapsed = time.perf_counter() - t0
    print(f"    {final['before']} → {final['after']} "
          f"(ratio {final['ratio']:.3f}) in {elapsed:.1f}s", flush=True)
    if final["new_rules"]:
        print(f"    Discovered {len(final['new_rules'])} rules:", flush=True)
        for r in final["new_rules"][:8]:
            print(f"      + {r}", flush=True)
        if len(final["new_rules"]) > 8:
            print(f"      ... and {len(final['new_rules'])-8} more", flush=True)

    # ── Summary table ────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  EVOLUTION TIMELINE")
    print(f"{'='*72}")
    print(f"  {'#':<3} {'Session':<40} {'Cls':>5} {'F':>4} {'R':>4} {'Drm':>6}")
    print(f"  {'-'*3} {'-'*40} {'-'*5} {'-'*4} {'-'*4} {'-'*6}")

    for s in timeline:
        dream_str = ""
        if "dream" in s:
            d = s["dream"]
            sign = "+" if d["removed"] < 0 else "-"
            dream_str = f"{sign}{abs(d['removed'])}"
        name = s["name"]
        if len(name) > 40:
            name = name[:37] + "..."
        print(f"  {s['session']:<3} {name:<40} {s['total_clauses']:>5} "
              f"{s['facts']:>4} {s['rules']:>4} {dream_str:>6}")

    # ── Final state ──────────────────────────────────────────────
    print(f"\n  Final KB: {len(store.kb)} clauses "
          f"({len(store.kb.facts)} facts, {len(store.kb.rules)} rules)")

    print(f"\n  Rules in final KB ({len(store.kb.rules)}):")
    for rule in store.kb.rules:
        print(f"    {rule}")

    # ── Budget ───────────────────────────────────────────────────
    if store.llm_client:
        u = store.llm_client.usage
        c = store.llm_client.estimated_cost()
        print(f"\n  LLM usage: {u.calls} calls, "
              f"{u.input_tokens:,} in / {u.output_tokens:,} out "
              f"(${c:.4f})")

    # ── Correctness spot-checks ──────────────────────────────────
    print(f"\n  Correctness spot-checks:")
    checks = [
        # Direct facts
        ("(advisor amir chen)", True, "direct fact"),
        ("(expertise chen ml)", True, "direct fact"),
        # User-provided rules
        ("(group_member amir)", True, "user rule: advisor→group_member"),
        ("(group_member bea)", True, "user rule"),
        ("(active_member elena)", True, "not alumni → active"),
        ("(active_member carlos)", False, "carlos is alumni"),
        ("(senior_student amir)", True, "passed quals"),
        ("(senior_student jun)", False, "jun hasn't passed quals"),
        ("(can_mentor amir)", True, "senior student can mentor"),
        ("(can_mentor hiro)", True, "postdoc can mentor"),
        # Transitive citation
        ("(indirectly_cites p007 p_ext1)", True, "p007→p001→p_ext1"),
        # Negative
        ("(advisor chen prof_kim)", False, "chen is not advised by kim"),
        ("(funded_by elena nsf_2301)", False, "elena is self-funded"),
    ]

    all_pass = True
    for query, expected, reason in checks:
        result = store.query(query)
        got = result["count"] > 0
        status = "PASS" if got == expected else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"    {status}: {query} = {got}  [{reason}]")

    print(f"\n  All pass: {all_pass}")

    # ── Predicate census ─────────────────────────────────────────
    functors = {}
    for f in store.kb.facts:
        if isinstance(f.term, Compound):
            functors[f.term.functor] = functors.get(f.term.functor, 0) + 1
    print(f"\n  Predicate census ({len(functors)} predicates):")
    for fn, count in sorted(functors.items(), key=lambda x: -x[1])[:20]:
        print(f"    {fn}: {count}")
    if len(functors) > 20:
        print(f"    ... and {len(functors) - 20} more predicates")

    # ── Save ─────────────────────────────────────────────────────
    timeline_path = store_path.with_suffix(".timeline.json")
    with open(timeline_path, "w") as f:
        json.dump(timeline, f, indent=2)
    print(f"\n  Timeline: {timeline_path}")

    return timeline


if __name__ == "__main__":
    run_simulation()
