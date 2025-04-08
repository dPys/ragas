from __future__ import annotations

import logging
import random
import os
import pickle
import typing as t
from dataclasses import dataclass, field

from langchain_core.callbacks import BaseCallbackManager
from langchain_core.documents import Document as LCDocument

from ragas._analytics import TestsetGenerationEvent, track
from ragas.callbacks import new_group
from ragas.cost import TokenUsageParser
from ragas.embeddings.base import (
    BaseRagasEmbeddings,
    LangchainEmbeddingsWrapper,
    LlamaIndexEmbeddingsWrapper,
)
from ragas.executor import Executor
from ragas.llms import BaseRagasLLM, LangchainLLMWrapper, LlamaIndexLLMWrapper
from ragas.run_config import RunConfig
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.persona import Persona, generate_personas_from_kg
from ragas.testset.synthesizers import default_query_distribution
from ragas.testset.synthesizers.testset_schema import Testset, TestsetSample
from ragas.testset.synthesizers.utils import calculate_split_values
from ragas.testset.synthesizers.single_hop.specific import SingleHopScenario
from ragas.testset.transforms import Transforms, apply_transforms, default_transforms

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks
    from langchain_core.embeddings import Embeddings as LangchainEmbeddings
    from langchain_core.language_models import BaseLanguageModel as LangchainLLM
    from llama_index.core.base.embeddings.base import (
        BaseEmbedding as LlamaIndexEmbedding,
    )
    from llama_index.core.base.llms.base import BaseLLM as LlamaIndexLLM
    from llama_index.core.schema import Document as LlamaIndexDocument

    from ragas.embeddings.base import BaseRagasEmbeddings
    from ragas.llms.base import BaseRagasLLM
    from ragas.testset.synthesizers import QueryDistribution
    from ragas.testset.synthesizers.base import BaseScenario


RAGAS_TESTSET_GENERATION_GROUP_NAME = "ragas testset generation"
DOCUMENT_ID_METADATA_KEYS = ['document_id', 'global_id', 'id']

logger = logging.getLogger(__name__)


@dataclass
class TestsetGenerator:
    """
    Generates an evaluation dataset based on given scenarios and parameters.

    Attributes
    ----------
    llm : BaseRagasLLM
        The language model to use for the generation process.
    knowledge_graph : KnowledgeGraph, default empty
        The knowledge graph to use for the generation process.
    """

    llm: BaseRagasLLM
    embedding_model: BaseRagasEmbeddings
    knowledge_graph: KnowledgeGraph = field(default_factory=KnowledgeGraph)
    persona_list: t.Optional[t.List[Persona]] = None

    @classmethod
    def from_langchain(
        cls,
        llm: LangchainLLM,
        embedding_model: LangchainEmbeddings,
        knowledge_graph: t.Optional[KnowledgeGraph] = None,
    ) -> TestsetGenerator:
        """
        Creates a `TestsetGenerator` from a Langchain LLMs.
        """
        knowledge_graph = knowledge_graph or KnowledgeGraph()
        return cls(
            LangchainLLMWrapper(llm),
            LangchainEmbeddingsWrapper(embedding_model),
            knowledge_graph,
        )

    @classmethod
    def from_llama_index(
        cls,
        llm: LlamaIndexLLM,
        embedding_model: LlamaIndexEmbedding,
        knowledge_graph: t.Optional[KnowledgeGraph] = None,
    ) -> TestsetGenerator:
        """
        Creates a `TestsetGenerator` from a LlamaIndex LLM and embedding model.
        """
        knowledge_graph = knowledge_graph or KnowledgeGraph()
        return cls(
            LlamaIndexLLMWrapper(llm),
            LlamaIndexEmbeddingsWrapper(embedding_model),
            knowledge_graph,
        )

    def generate_with_langchain_docs(
        self,
        documents: t.Sequence[LCDocument],
        testset_size: int,
        transforms: t.Optional[Transforms] = None,
        transforms_llm: t.Optional[BaseRagasLLM] = None,
        transforms_embedding_model: t.Optional[BaseRagasEmbeddings] = None,
        query_distribution: t.Optional[QueryDistribution] = None,
        run_config: t.Optional[RunConfig] = None,
        callbacks: t.Optional[Callbacks] = None,
        with_debugging_logs=False,
        raise_exceptions: bool = True,
    ) -> Testset:
        """
        Generates an evaluation dataset based on given Langchain documents and parameters.

        Parameters
        ----------
        documents : Sequence[LCDocument]
            A sequence of Langchain documents to use as source material
        testset_size : int
            The number of test samples to generate
        transforms : Optional[Transforms], optional
            Custom transforms to apply to the documents, by default None
        transforms_llm : Optional[BaseRagasLLM], optional
            LLM to use for transforms if different from instance LLM, by default None
        transforms_embedding_model : Optional[BaseRagasEmbeddings], optional
            Embedding model to use for transforms if different from instance model, by default None
        query_distribution : Optional[QueryDistribution], optional
            Distribution of query types to generate, by default None
        run_config : Optional[RunConfig], optional
            Configuration for the generation run, by default None
        callbacks : Optional[Callbacks], optional
            Callbacks to use during generation, by default None
        with_debugging_logs : bool, optional
            Whether to include debug logs, by default False
        raise_exceptions : bool, optional
            Whether to raise exceptions during generation, by default True

        Returns
        -------
        Testset
            The generated evaluation dataset

        Raises
        ------
        ValueError
            If no LLM or embedding model is provided either during initialization or as arguments
        """

        # force the user to provide an llm and embedding client to prevent use of default LLMs
        if not self.llm and not transforms_llm:
            raise ValueError(
                """An llm client was not provided.
                       Provide an LLM on TestsetGenerator instantiation or as an argument for transforms_llm parameter.
                       Alternatively you can provide your own transforms through the `transforms` parameter."""
            )
        if not self.embedding_model and not transforms_embedding_model:
            raise ValueError(
                """An embedding client was not provided. Provide an embedding through the transforms_embedding_model parameter. Alternatively you can provide your own transforms through the `transforms` parameter."""
            )

        if not transforms:
            transforms = default_transforms(
                documents=list(documents),
                llm=transforms_llm or self.llm,
                embedding_model=transforms_embedding_model or self.embedding_model,
            )

        # convert the documents to Ragas nodes
        nodes = []
        for doc in documents:
            node = Node(
                type=NodeType.DOCUMENT,
                properties={
                    "page_content": doc.page_content,
                    "document_metadata": doc.metadata,
                },
            )
            nodes.append(node)

        kg = KnowledgeGraph(nodes=nodes)

        # apply transforms and update the knowledge graph
        apply_transforms(kg, transforms)
        self.knowledge_graph = kg

        return self.generate(
            testset_size=testset_size,
            query_distribution=query_distribution,
            run_config=run_config,
            callbacks=callbacks,
            with_debugging_logs=with_debugging_logs,
            raise_exceptions=raise_exceptions,
        )

    def generate_with_llamaindex_docs(
        self,
        documents: t.Sequence[LlamaIndexDocument],
        testset_size: int,
        transforms: t.Optional[Transforms] = None,
        transforms_llm: t.Optional[LlamaIndexLLM] = None,
        transforms_embedding_model: t.Optional[LlamaIndexEmbedding] = None,
        query_distribution: t.Optional[QueryDistribution] = None,
        run_config: t.Optional[RunConfig] = None,
        callbacks: t.Optional[Callbacks] = None,
        with_debugging_logs=False,
        raise_exceptions: bool = True,
    ):
        """
        Generates an evaluation dataset based on given scenarios and parameters.
        """

        run_config = run_config or RunConfig()

        # force the user to provide an llm and embedding client to prevent use of default LLMs
        if not self.llm and not transforms_llm:
            raise ValueError(
                "An llm client was not provided. Provide an LLM on TestsetGenerator instantiation or as an argument for transforms_llm parameter. Alternatively you can provide your own transforms through the `transforms` parameter."
            )
        if not self.embedding_model and not transforms_embedding_model:
            raise ValueError(
                "An embedding client was not provided. Provide an embedding through the transforms_embedding_model parameter. Alternatively you can provide your own transforms through the `transforms` parameter."
            )

        if not transforms:
            # use TestsetGenerator's LLM and embedding model if no transforms_llm or transforms_embedding_model is provided
            if transforms_llm is None:
                llm_for_transforms = self.llm
            else:
                llm_for_transforms = LlamaIndexLLMWrapper(transforms_llm)
            if transforms_embedding_model is None:
                embedding_model_for_transforms = self.embedding_model
            else:
                embedding_model_for_transforms = LlamaIndexEmbeddingsWrapper(
                    transforms_embedding_model
                )

            # create the transforms
            transforms = default_transforms(
                documents=[LCDocument(page_content=doc.text) for doc in documents],
                llm=llm_for_transforms,
                embedding_model=embedding_model_for_transforms,
            )

        # convert the documents to Ragas nodes
        nodes = []
        for doc in documents:
            if doc.text is not None and doc.text.strip() != "":
                node = Node(
                    type=NodeType.DOCUMENT,
                    properties={
                        "page_content": doc.text,
                        "document_metadata": doc.metadata,
                    },
                )
                nodes.append(node)

        kg = KnowledgeGraph(nodes=nodes)

        # apply transforms and update the knowledge graph
        apply_transforms(kg, transforms, run_config)
        self.knowledge_graph = kg

        return self.generate(
            testset_size=testset_size,
            query_distribution=query_distribution,
            run_config=run_config,
            callbacks=callbacks,
            with_debugging_logs=with_debugging_logs,
            raise_exceptions=raise_exceptions,
        )

    def generate(
        self: TestsetGenerator,
        testset_size: int,
        query_distribution: t.Optional[t.List[t.Tuple[t.Any, float]]] = None,
        num_personas: int = 3,
        run_config: t.Optional[RunConfig] = None,
        batch_size: t.Optional[int] = None,
        callbacks: t.Optional[t.Any] = None,
        token_usage_parser: t.Optional[t.Any] = None,
        with_debugging_logs=False,
        raise_exceptions: bool = False,
        scenario_cache_file: t.Optional[str] = None,
    ) -> Testset:
        """ Patched generate with detailed logging and error handling for scenario results. """
        if run_config is not None: self.llm.set_run_config(run_config)
        if query_distribution is None:
            query_distribution = default_query_distribution(self.llm, self.knowledge_graph)
        query_distribution_list = query_distribution
        callbacks = callbacks or []
        ragas_callbacks = {}
        if token_usage_parser is not None:
            from ragas.cost import CostCallbackHandler
            cost_cb = CostCallbackHandler(token_usage_parser=token_usage_parser)
            ragas_callbacks["cost_cb"] = cost_cb
        else: cost_cb = None
        for cb_instance in ragas_callbacks.values():
            if hasattr(callbacks, 'add_handler'):
                if cb_instance not in callbacks.handlers: callbacks.add_handler(cb_instance)
            elif isinstance(callbacks, list):
                if cb_instance not in callbacks: callbacks.append(cb_instance)

        testset_generation_rm, testset_generation_grp = new_group("ragas testset generation", inputs={"testset_size": testset_size}, callbacks=callbacks)
        if with_debugging_logs:
            logger.setLevel(logging.DEBUG)
            logging.getLogger("ragas.testset.graph").setLevel(logging.DEBUG)
            logging.getLogger("ragas.testset.transforms").setLevel(logging.DEBUG)

        personas_to_use = []
        active_persona_list = self.persona_list
        if active_persona_list is None:
            logger.info(f"Generating {num_personas} personas...")
            try:
                active_persona_list = generate_personas_from_kg(llm=self.llm, kg=self.knowledge_graph, num_personas=num_personas, callbacks=callbacks)
                self.persona_list = active_persona_list
                logger.info(f"Generated {len(active_persona_list)} personas.")
            except Exception as persona_err:
                logger.error(f"Error during persona generation: {persona_err}", exc_info=True)
                if raise_exceptions: raise persona_err
                active_persona_list = [] # Continue without personas if not raising
        else:
            logger.info(f"Using provided persona list ({len(active_persona_list)} personas).")


        if active_persona_list:
            persona_list_mutable = list(active_persona_list)
            random.shuffle(persona_list_mutable)
            personas_to_use = persona_list_mutable[:num_personas] # Use up to num_personas

        scenario_sample_list = None
        if os.path.exists(scenario_cache_file):
            try:
                print(f"Loading scenarios from cache: {scenario_cache_file}")
                with open(scenario_cache_file, 'rb') as f:
                    scenario_data_list = pickle.load(f)
                scenario_sample_list = []
                for synth_scenarios_data in scenario_data_list:
                    current_synth_scenarios = []
                    for scenario_dict in synth_scenarios_data:
                        rehydrated_scenario = SingleHopScenario(**scenario_dict.__dict__)
                        current_synth_scenarios.append(rehydrated_scenario)
                    scenario_sample_list.append(current_synth_scenarios)
                print("Scenarios loaded successfully from cache.")
                if not isinstance(scenario_sample_list, list):
                    print("WARNING: Cached data is not a list. Regenerating scenarios.")
                    scenario_sample_list = None
                elif scenario_sample_list and not isinstance(scenario_sample_list[0], list):
                    print("WARNING: Cached data structure seems incorrect (expected list of lists). Regenerating scenarios.")
                    scenario_sample_list = None

            except (pickle.UnpicklingError, EOFError, AttributeError, ImportError, IndexError) as e:
                print(f"Error loading cache file: {e}. Regenerating scenarios.")
                scenario_sample_list = None
                # if os.path.exists(scenario_cache_file):
                #      os.remove(scenario_cache_file)

        if scenario_sample_list is None:
            print("Generating scenarios (cache not found or invalid)...")
            splits, _ = calculate_split_values([prob for _, prob in query_distribution_list], testset_size)
            scenario_generation_rm, scenario_generation_grp = new_group("Scenario Generation", inputs={"splits": splits}, callbacks=testset_generation_grp)

            scenario_raise_exceptions = True

            exec_scenarios = Executor(
                "Generating Scenarios",
                raise_exceptions=scenario_raise_exceptions,
                run_config=run_config,
                keep_progress_bar=False,
                batch_size=batch_size
            )
            for i, (scenario_generator, _) in enumerate(query_distribution_list):
                exec_scenarios.submit(scenario_generator.generate_scenarios, n=splits[i], knowledge_graph=self.knowledge_graph, persona_list=personas_to_use, callbacks=scenario_generation_grp)

            try:
                scenario_sample_list = exec_scenarios.results()
                scenario_generation_rm.on_chain_end(outputs={"scenario_sample_list": scenario_sample_list})

                try:
                    print(f"Saving scenarios to cache: {scenario_cache_file}")
                    with open(scenario_cache_file, 'wb') as f:
                        pickle.dump(scenario_sample_list, f)
                    print("Scenarios saved successfully.")
                except Exception as e:
                    print(f"Error saving scenarios to cache: {e}")

            except Exception as e:
                scenario_generation_rm.on_chain_error(e)
                print(f"FATAL: Error during scenario generation: {e}")
                if scenario_raise_exceptions:
                    raise e
                else:
                    scenario_sample_list = []

            if not isinstance(scenario_sample_list, list):
                print(f"ERROR: Scenario generation result is not a list ({type(scenario_sample_list)}). Setting to empty list.")
                scenario_sample_list = []

        sample_generation_rm, sample_generation_grp = new_group("Sample Generation", inputs={"scenario_sample_list": "Loaded from cache" if os.path.exists(scenario_cache_file) else "Newly generated"}, callbacks=testset_generation_grp)

        exec_samples = Executor(
            "Generating Samples",
            raise_exceptions=False,
            run_config=run_config,
            keep_progress_bar=True,
            batch_size=batch_size
        )

        sample_generation_rm, sample_generation_grp = new_group("Sample Generation", inputs={"scenario_sample_list": scenario_sample_list}, callbacks=testset_generation_grp)
        exec_samples = Executor("Generating Samples", raise_exceptions=raise_exceptions, run_config=run_config, keep_progress_bar=True, batch_size=batch_size)
        tasks_to_submit = []
        additional_info_ordered = []

        logger.info("Preparing samples based on generated scenarios...")
        for i, (synthesizer, _) in enumerate(query_distribution_list):
            if i < len(scenario_sample_list):
                current_scenario_results = scenario_sample_list[i]

                if not isinstance(current_scenario_results, list):
                    logger.error(f"Executor returned non-list result for synthesizer {i} ({synthesizer.name}). Type: {type(current_scenario_results)}. Value: {current_scenario_results}. Skipping sample generation for this batch.")
                    num_expected_samples = splits[i]
                    for _ in range(num_expected_samples):
                        additional_info_ordered.append({"synthesizer_name": synthesizer.name, "document_id": None, "error": "Scenario generation failed"})
                    continue

                if not current_scenario_results:
                    logger.warning(f"Synthesizer {i} ({synthesizer.name}) generated an empty list of scenarios. No samples will be generated for this batch.")
                    num_expected_samples = splits[i]
                    for _ in range(num_expected_samples):
                        additional_info_ordered.append({"synthesizer_name": synthesizer.name, "document_id": None, "error": "No scenarios generated"})

                for idx, scenario in enumerate(current_scenario_results):
                    if scenario is None:
                        logger.warning(f"Encountered None scenario at index {idx} for synthesizer {i}. Skipping.")
                        additional_info_ordered.append({"synthesizer_name": synthesizer.name, "document_id": None, "error": "Invalid scenario object"})
                        continue

                    source_doc_id = None
                    source_node = None
                    try:
                        if hasattr(scenario, 'source_node') and scenario.source_node:
                            source_node = scenario.source_node
                        elif hasattr(scenario, 'nodes') and scenario.nodes:
                            if scenario.nodes and all(isinstance(n, Node) for n in scenario.nodes):
                                source_node = next((n for n in scenario.nodes if n.type == NodeType.DOCUMENT), None)
                            else:
                                logger.warning(f"Scenario {i}-{idx} has invalid 'nodes' attribute: {scenario.nodes}")
                        else:
                            logger.warning(f"Scenario {i}-{idx} has no 'source_node' or 'nodes' attribute.")


                        if source_node and isinstance(source_node, Node) and source_node.type == NodeType.DOCUMENT: # Added isinstance check
                            metadata_found = source_node.properties.get("document_metadata")
                            if metadata_found and isinstance(metadata_found, dict):
                                for key in DOCUMENT_ID_METADATA_KEYS:
                                    source_doc_id = metadata_found.get(key)
                                    if source_doc_id is not None:
                                        source_doc_id = str(source_doc_id)
                                        break # Found ID
                            # else:
                            #      logger.debug(f"Scenario {i}-{idx}: No document_metadata dict found on source_node.")
                        # elif source_node:
                        #       logger.debug(f"Scenario {i}-{idx}: Source node found but is not NodeType.DOCUMENT or not a Node object.")
                        # else:
                        #      logger.debug(f"Scenario {i}-{idx}: No source node found.")


                    except Exception as e:
                        logger.warning(f"Error extracting document_id for scenario {i}-{idx}: {e}", exc_info=True)

                    info = {"synthesizer_name": synthesizer.name, "document_id": source_doc_id}
                    if source_doc_id is None:
                        info["error"] = "Document ID not extracted"
                    additional_info_ordered.append(info)

                    if hasattr(synthesizer, 'generate_sample'):
                        tasks_to_submit.append((synthesizer.generate_sample, {"scenario": scenario, "callbacks": sample_generation_grp}))
                    else:
                        logger.warning(f"Synthesizer {synthesizer.name} missing generate_sample method. Cannot submit task for scenario {i}-{idx}.")
            else:
                if i < len(splits):
                    logger.warning(f"Scenario results missing for synthesizer index {i} ({synthesizer.name}). Expected {splits[i]} scenarios.")
                    num_expected_samples = splits[i]
                    for _ in range(num_expected_samples):
                        additional_info_ordered.append({"synthesizer_name": synthesizer.name, "document_id": None, "error": "Scenario results missing"})
                else:
                    logger.error(f"Index {i} out of bounds for both scenario_sample_list and splits.")


        eval_samples = []
        if tasks_to_submit:
            logger.info(f"Submitting {len(tasks_to_submit)} sample generation tasks...")
            for func, kwargs in tasks_to_submit: exec_samples.submit(func, **kwargs)
            try:
                eval_samples = exec_samples.results()
            except Exception as e:
                sample_generation_rm.on_chain_error(e)
                if raise_exceptions: raise e
            else:
                sample_generation_rm.on_chain_end(outputs={"eval_samples": eval_samples})
                logger.info(f"Received {len(eval_samples)} results from sample generation.")
        else:
            logger.warning("No valid sample generation tasks were submitted.")
            sample_generation_rm.on_chain_end(outputs={"eval_samples": []})


        testset_samples = []
        logger.info("Constructing final Testset...")

        num_results_expected = len(additional_info_ordered)
        num_results_actual = len(eval_samples)

        if num_results_actual != num_results_expected:
            logger.error(f"Critical mismatch: Expected {num_results_expected} results based on scenarios/info, but got {num_results_actual} eval_samples. Testset will likely be incomplete or misaligned.")
            min_len = min(num_results_actual, num_results_expected)
            processed_indices = 0
            aligned_samples = []
            aligned_info = []

            eval_sample_iter = iter(eval_samples)
            current_eval_sample = next(eval_sample_iter, StopIteration)

            for idx, info in enumerate(additional_info_ordered):
                if info.get("error"):
                    continue

                if current_eval_sample is StopIteration:
                    logger.warning(f"Ran out of eval_samples while processing info index {idx}. Stopping alignment.")
                    break

                aligned_samples.append(current_eval_sample)
                aligned_info.append(info)
                current_eval_sample = next(eval_sample_iter, StopIteration) # Move to next sample

            final_eval_samples = aligned_samples
            final_additional_info = aligned_info
            logger.warning(f"Attempted alignment: Resulting in {len(final_eval_samples)} matched samples.")

        else:
            final_eval_samples = []
            final_additional_info = []
            for i, (sample, info) in enumerate(zip(eval_samples, additional_info_ordered)):
                if sample is None:
                    logger.warning(f"Sample at index {i} is None (likely due to sample generation error). Skipping.")
                    continue
                final_eval_samples.append(sample)
                final_additional_info.append(info)

        for i, (eval_sample, info) in enumerate(zip(final_eval_samples, final_additional_info)):
            doc_id_to_assign = info.get("document_id")
            synthesizer_name = info.get("synthesizer_name", "unknown")

            if not hasattr(eval_sample, '__dict__'):
                logger.error(f"Invalid eval_sample structure at index {i}: {eval_sample}. Skipping TestsetSample creation.")
                continue

            try:
                ts_sample = TestsetSample(
                    eval_sample=eval_sample,
                    synthesizer_name=synthesizer_name
                )
                setattr(ts_sample, 'document_id', doc_id_to_assign)
                testset_samples.append(ts_sample)
            except Exception as creation_error:
                logger.error(f"ERROR creating/assigning TestsetSample {i}: {creation_error}", exc_info=True)

        logger.info(f"Successfully created {len(testset_samples)} TestsetSample objects.")
        testset = Testset(samples=testset_samples, cost_cb=cost_cb)
        testset_generation_rm.on_chain_end({"testset": testset})

        try:
            from ragas._analytics import TestsetGenerationEvent, track
            track(
                TestsetGenerationEvent(
                    event_type="testset_generation",
                    evolution_names=[e.__class__.__name__.lower() for e, _ in query_distribution_list],
                    evolution_percentages=[p for _, p in query_distribution_list],
                    num_rows=len(testset_samples),
                    language="english",
                )
            )
        except ImportError:
            logger.debug("Analytics tracking skipped (module not found).")
        except Exception as track_err:
            logger.warning(f"Analytics tracking failed: {track_err}")

        return testset
