def initialize_knowledge_filters(knowledge_base):
    """Initialize valid metadata filters from existing vector database data to fix Agno's limitation on invalid filter keys.

    Args:
        knowledge_base: The AgentKnowledge instance to initialize filters for
    """
    try:
        # Manually set the valid metadata filters based on our known structure
        if knowledge_base.valid_metadata_filters is None:
            knowledge_base.valid_metadata_filters = set()

        # Add the agent_knowledge table filters
        knowledge_base.valid_metadata_filters.add("user_id")

    except Exception as e:
        print(f"Failed to initialize knowledge filters: {str(e)}")
