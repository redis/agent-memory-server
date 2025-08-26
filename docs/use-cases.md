# Use Cases and Examples

Redis Agent Memory Server enables powerful AI applications by providing persistent, searchable memory. Here are real-world use cases and implementation examples across different industries and applications.

## Customer Support

### Intelligent Support Agent

**Challenge**: Support agents need context about customer history, preferences, and previous issues to provide personalized assistance.

**Solution**: Memory server stores customer interactions, preferences, and issue history for instant retrieval.

```python
from agent_memory_client import MemoryAPIClient

client = MemoryAPIClient(base_url="http://localhost:8000")

# Store customer profile and preferences
await client.create_long_term_memories([
    {
        "text": "Customer Alice Johnson (alice@company.com) prefers email communication, has Pro subscription, works in marketing team",
        "memory_type": "semantic",
        "topics": ["customer_profile", "communication", "subscription"],
        "entities": ["Alice Johnson", "alice@company.com", "Pro subscription", "marketing"],
        "user_id": "alice_johnson",
        "namespace": "customer_support"
    }
])

# Store previous issue resolution
await client.create_long_term_memories([
    {
        "text": "Customer Alice Johnson resolved login issue on January 10, 2024 by clearing browser cache",
        "memory_type": "episodic",
        "event_date": "2024-01-10T14:30:00Z",
        "topics": ["support_ticket", "login_issue", "resolution"],
        "entities": ["login issue", "browser cache", "resolved"],
        "user_id": "alice_johnson",
        "namespace": "customer_support"
    }
])

# Later, when customer contacts support again
async def handle_support_request(customer_email: str, current_issue: str):
    # Search for customer history
    customer_context = await client.search_long_term_memory(
        text=f"customer {customer_email} preferences issues history",
        filters={"user_id": {"eq": customer_email.replace("@", "_")}},
        limit=5
    )

    # Generate contextual response
    context_prompt = await client.memory_prompt(
        query=f"Customer is reporting: {current_issue}",
        session_id=f"support_{customer_email}",
        long_term_search={
            "text": f"{current_issue} similar issues resolution",
            "limit": 3
        }
    )

    return context_prompt  # Use with your AI model
```

**Benefits**:
- Personalized support based on customer history
- Faster issue resolution with similar case lookup
- Consistent support across different agents
- Automatic learning from resolved cases

## Personal AI Assistant

### Proactive Personal Memory

**Challenge**: Users want an AI assistant that remembers their preferences, schedules, and context across conversations spanning days or weeks.

**Solution**: Dual-layer memory system that maintains conversation state and learns long-term preferences.

```python
class PersonalAssistant:
    def __init__(self):
        self.client = MemoryAPIClient(base_url="http://localhost:8000")
        self.user_id = "user_john_doe"

    async def process_conversation_turn(self, session_id: str, user_message: str, assistant_response: str):
        # Store conversation in working memory
        working_memory = WorkingMemory(
            session_id=session_id,
            messages=[
                MemoryMessage(role="user", content=user_message),
                MemoryMessage(role="assistant", content=assistant_response)
            ],
            user_id=self.user_id
        )

        # System automatically extracts important information to long-term memory
        await self.client.set_working_memory(session_id, working_memory)

    async def get_contextual_response(self, session_id: str, user_query: str):
        # Get enriched prompt with personal context
        prompt_data = await self.client.memory_prompt(
            query=user_query,
            session={
                "session_id": session_id,
                "user_id": self.user_id,
                "model_name": "gpt-4o"
            },
            long_term_search={
                "text": user_query,
                "filters": {"user_id": {"eq": self.user_id}},
                "limit": 5,
                "recency_boost": True  # Prefer recent relevant memories
            }
        )

        # Send to your AI model for contextual response
        return prompt_data

# Example conversation flow
assistant = PersonalAssistant()

# Day 1: User shares preferences
await assistant.process_conversation_turn(
    session_id="daily_chat_2024_01_15",
    user_message="I prefer meetings in the morning and I'm vegetarian",
    assistant_response="I'll remember your morning meeting preference and dietary restrictions!"
)

# Week later: Assistant uses stored context
response_data = await assistant.get_contextual_response(
    session_id="daily_chat_2024_01_22",
    user_query="Can you help me plan lunch for my team meeting?"
)
# AI model will have context about user being vegetarian and preferring morning meetings
```

**Benefits**:
- Conversations that span weeks maintain context
- Learning user preferences and patterns over time
- Proactive suggestions based on historical data
- Seamless experience across different devices/sessions

## Software Development

### Intelligent Code Assistant

**Challenge**: Developers need an AI assistant that understands their codebase, coding patterns, and project context to provide relevant help.

**Solution**: Memory system stores project context, coding patterns, and problem-solution pairs for contextual assistance.

```python
class CodingAssistant:
    def __init__(self, project_name: str):
        self.client = MemoryAPIClient(base_url="http://localhost:8000")
        self.project_namespace = f"project_{project_name}"

    async def learn_project_context(self):
        """Store project architecture and patterns"""
        project_memories = [
            {
                "text": "Project uses FastAPI with SQLAlchemy ORM, PostgreSQL database, and Redis for caching",
                "memory_type": "semantic",
                "topics": ["architecture", "tech_stack", "fastapi", "sqlalchemy", "postgresql"],
                "entities": ["FastAPI", "SQLAlchemy", "PostgreSQL", "Redis"],
                "namespace": self.project_namespace
            },
            {
                "text": "Database migrations managed with Alembic, models defined in app/models/ directory",
                "memory_type": "semantic",
                "topics": ["database", "migrations", "models", "alembic"],
                "entities": ["Alembic", "app/models", "database migrations"],
                "namespace": self.project_namespace
            },
            {
                "text": "Authentication uses JWT tokens with 24-hour expiry, implemented in app/auth.py",
                "memory_type": "semantic",
                "topics": ["authentication", "jwt", "security"],
                "entities": ["JWT tokens", "app/auth.py", "authentication"],
                "namespace": self.project_namespace
            }
        ]

        await self.client.create_long_term_memories(project_memories)

    async def store_solution_pattern(self, problem: str, solution: str, code_example: str = None):
        """Store problem-solution patterns for reuse"""
        memory_text = f"Problem: {problem}\nSolution: {solution}"
        if code_example:
            memory_text += f"\nCode example: {code_example}"

        await self.client.create_long_term_memories([{
            "text": memory_text,
            "memory_type": "episodic",
            "topics": ["problem_solving", "code_patterns", "solutions"],
            "entities": [problem, solution],
            "namespace": self.project_namespace
        }])

    async def get_contextual_help(self, coding_question: str):
        """Get help with project-specific context"""
        return await self.client.memory_prompt(
            query=coding_question,
            long_term_search={
                "text": coding_question,
                "filters": {"namespace": {"eq": self.project_namespace}},
                "limit": 3
            }
        )

# Usage example
assistant = CodingAssistant("ecommerce_api")

# Setup project context
await assistant.learn_project_context()

# Store a solved problem
await assistant.store_solution_pattern(
    problem="Database connection pooling issues under high load",
    solution="Configure SQLAlchemy pool_size=20, max_overflow=0, pool_pre_ping=True",
    code_example="engine = create_engine(url, pool_size=20, max_overflow=0, pool_pre_ping=True)"
)

# Later, get contextual help
help_data = await assistant.get_contextual_help(
    "How do I optimize database connections for better performance?"
)
# AI model will have context about the FastAPI + SQLAlchemy setup and previous solutions
```

**Benefits**:
- Project-specific advice based on actual tech stack
- Learn from previous solutions and code patterns
- Maintain context about architectural decisions
- Accelerate development with contextual suggestions

## Content and Research

### Research Assistant with Memory

**Challenge**: Researchers need to synthesize information across multiple sources, sessions, and topics while maintaining context and preventing information loss.

**Solution**: Structured memory system that organizes research findings, tracks sources, and maintains topic relationships.

```python
class ResearchAssistant:
    def __init__(self, research_topic: str):
        self.client = MemoryAPIClient(base_url="http://localhost:8000")
        self.topic_namespace = f"research_{research_topic.lower().replace(' ', '_')}"

    async def store_research_finding(self, finding: str, source: str, topics: list,
                                   confidence: str = "high", date_found: str = None):
        """Store research findings with metadata"""
        memory_text = f"Finding: {finding}\nSource: {source}\nConfidence: {confidence}"

        await self.client.create_long_term_memories([{
            "text": memory_text,
            "memory_type": "episodic" if date_found else "semantic",
            "event_date": date_found,
            "topics": topics + ["research_finding", confidence],
            "entities": [source, finding[:50] + "..."],
            "namespace": self.topic_namespace
        }])

    async def synthesize_knowledge(self, research_question: str):
        """Find related knowledge for synthesis"""
        related_findings = await self.client.search_long_term_memory(
            text=research_question,
            filters={"namespace": {"eq": self.topic_namespace}},
            limit=10,
            recency_boost=True
        )

        # Generate synthesis prompt
        synthesis_prompt = await self.client.memory_prompt(
            query=f"Synthesize research findings to answer: {research_question}",
            long_term_search={
                "text": research_question,
                "filters": {"namespace": {"eq": self.topic_namespace}},
                "limit":8
            }
        )

        return synthesis_prompt

    async def track_research_gaps(self, completed_areas: list, remaining_questions: list):
        """Track research progress and gaps"""
        progress_memory = {
            "text": f"Research progress - Completed: {', '.join(completed_areas)}. "
                   f"Remaining questions: {', '.join(remaining_questions)}",
            "memory_type": "episodic",
            "event_date": datetime.now().isoformat(),
            "topics": ["research_progress", "gaps", "planning"],
            "entities": completed_areas + remaining_questions,
            "namespace": self.topic_namespace
        }

        await self.client.create_long_term_memories([progress_memory])

# Usage example
research = ResearchAssistant("AI Memory Systems")

# Store findings from different sources
await research.store_research_finding(
    finding="Vector databases show 40% better performance for semantic search compared to traditional keyword search",
    source="IEEE AI Conference 2024, Smith et al.",
    topics=["performance", "vector_search", "benchmarks"],
    confidence="high",
    date_found="2024-01-15T10:00:00Z"
)

await research.store_research_finding(
    finding="Memory consolidation in AI systems improves long-term retention by 60%",
    source="Nature AI Journal, Vol 15, Johnson & Lee",
    topics=["memory_consolidation", "retention", "performance"],
    confidence="high"
)

# Later, synthesize knowledge
synthesis = await research.synthesize_knowledge(
    "What are the performance benefits of advanced memory systems in AI applications?"
)
# AI model will have access to relevant findings from multiple sources
```

**Benefits**:
- Organize research findings across multiple sessions
- Synthesize knowledge from diverse sources
- Track research progress and identify gaps
- Maintain source attribution and confidence levels

## E-Commerce and Retail

### Personalized Shopping Assistant

**Challenge**: E-commerce platforms need to provide personalized recommendations based on browsing history, purchase patterns, and stated preferences across multiple sessions.

**Solution**: Memory system tracks user preferences, purchase history, and contextual shopping behavior.

```python
class ShoppingAssistant:
    def __init__(self):
        self.client = MemoryAPIClient(base_url="http://localhost:8000")

    async def track_browsing_behavior(self, user_id: str, product_category: str,
                                    products_viewed: list, time_spent: int):
        """Store browsing patterns"""
        await self.client.create_long_term_memories([{
            "text": f"User spent {time_spent} minutes browsing {product_category}, "
                   f"viewed {len(products_viewed)} products: {', '.join(products_viewed[:3])}",
            "memory_type": "episodic",
            "event_date": datetime.now().isoformat(),
            "topics": ["browsing", product_category, "shopping_behavior"],
            "entities": products_viewed + [product_category],
            "user_id": user_id,
            "namespace": "ecommerce"
        }])

    async def store_purchase(self, user_id: str, products: list, total_amount: float,
                           occasion: str = None):
        """Store purchase history"""
        memory_text = f"Purchase: {', '.join(products)} for ${total_amount:.2f}"
        if occasion:
            memory_text += f" for {occasion}"

        await self.client.create_long_term_memories([{
            "text": memory_text,
            "memory_type": "episodic",
            "event_date": datetime.now().isoformat(),
            "topics": ["purchase", "transaction"] + ([occasion] if occasion else []),
            "entities": products + [f"${total_amount:.2f}"],
            "user_id": user_id,
            "namespace": "ecommerce"
        }])

    async def store_preferences(self, user_id: str, preferences: dict):
        """Store explicit user preferences"""
        for category, preference in preferences.items():
            await self.client.create_long_term_memories([{
                "text": f"User prefers {preference} in {category} category",
                "memory_type": "semantic",
                "topics": ["preferences", category],
                "entities": [preference, category],
                "user_id": user_id,
                "namespace": "ecommerce"
            }])

    async def get_personalized_recommendations(self, user_id: str, current_context: str):
        """Generate personalized recommendations"""
        recommendation_prompt = await self.client.memory_prompt(
            query=f"Recommend products for user context: {current_context}",
            long_term_search={
                "text": f"{current_context} preferences purchases browsing",
                "filters": {"user_id": {"eq": user_id}, "namespace": {"eq": "ecommerce"}},
                "limit": 5,
                "recency_boost": True
            }
        )

        return recommendation_prompt

# Usage example
shopping = ShoppingAssistant()
user_id = "customer_jane_smith"

# Track user behavior over time
await shopping.track_browsing_behavior(
    user_id=user_id,
    product_category="outdoor_gear",
    products_viewed=["hiking_boots_model_x", "waterproof_jacket_y", "camping_tent_z"],
    time_spent=25
)

await shopping.store_purchase(
    user_id=user_id,
    products=["hiking_boots_model_x", "wool_socks"],
    total_amount=149.99,
    occasion="upcoming_hiking_trip"
)

await shopping.store_preferences(
    user_id=user_id,
    preferences={
        "brands": "sustainable and eco-friendly brands",
        "price_range": "mid-range products ($50-$200)",
        "style": "functional outdoor gear with minimal design"
    }
)

# Later, generate personalized recommendations
recommendations = await shopping.get_personalized_recommendations(
    user_id=user_id,
    current_context="looking for winter outdoor gear"
)
# AI will have context about hiking interests, eco-friendly preference, price range, etc.
```

**Benefits**:
- Personalized recommendations based on complete user history
- Cross-session shopping context and preferences
- Seasonal and contextual product suggestions
- Improved conversion through relevant recommendations

## Education and Training

### Adaptive Learning Assistant

**Challenge**: Educational platforms need to track student progress, identify knowledge gaps, and provide personalized learning paths.

**Solution**: Memory system tracks learning progress, concept understanding, and adapts instruction based on individual needs.

```python
class LearningAssistant:
    def __init__(self, course_id: str):
        self.client = MemoryAPIClient(base_url="http://localhost:8000")
        self.course_namespace = f"course_{course_id}"

    async def track_concept_understanding(self, student_id: str, concept: str,
                                        understanding_level: str, evidence: str):
        """Track student understanding of concepts"""
        await self.client.create_long_term_memories([{
            "text": f"Student understanding of {concept}: {understanding_level}. "
                   f"Evidence: {evidence}",
            "memory_type": "episodic",
            "event_date": datetime.now().isoformat(),
            "topics": ["learning_progress", concept, understanding_level],
            "entities": [concept, understanding_level],
            "user_id": student_id,
            "namespace": self.course_namespace
        }])

    async def store_learning_preference(self, student_id: str, preference_type: str,
                                      preference: str):
        """Store individual learning preferences"""
        await self.client.create_long_term_memories([{
            "text": f"Student learns best through {preference} for {preference_type}",
            "memory_type": "semantic",
            "topics": ["learning_style", preference_type],
            "entities": [preference, preference_type],
            "user_id": student_id,
            "namespace": self.course_namespace
        }])

    async def identify_knowledge_gaps(self, student_id: str, topic_area: str):
        """Identify areas where student needs help"""
        progress_search = await self.client.search_long_term_memory(
            text=f"{topic_area} understanding progress",
            filters={
                "user_id": {"eq": student_id},
                "namespace": {"eq": self.course_namespace}
            },
            limit=10
        )

        # Find concepts with low understanding
        weak_concepts = []
        for memory in progress_search.memories:
            if "struggling" in memory.text.lower() or "confused" in memory.text.lower():
                weak_concepts.append(memory)

        return weak_concepts

    async def generate_personalized_instruction(self, student_id: str,
                                              target_concept: str):
        """Generate personalized instruction based on student history"""
        instruction_prompt = await self.client.memory_prompt(
            query=f"Create personalized instruction for {target_concept}",
            long_term_search={
                "text": f"{target_concept} learning style preferences understanding",
                "filters": {
                    "user_id": {"eq": student_id},
                    "namespace": {"eq": self.course_namespace}
                },
                "limit": 5
            }
        )

        return instruction_prompt

# Usage example
learning = LearningAssistant("python_programming_101")
student_id = "student_alex_chen"

# Track learning progress
await learning.track_concept_understanding(
    student_id=student_id,
    concept="object_oriented_programming",
    understanding_level="struggling",
    evidence="Had difficulty with inheritance exercise, asked 3 questions about method overriding"
)

await learning.store_learning_preference(
    student_id=student_id,
    preference_type="explanation_style",
    preference="visual diagrams and concrete examples rather than abstract theory"
)

# Identify knowledge gaps
gaps = await learning.identify_knowledge_gaps(student_id, "object_oriented_programming")

# Generate personalized instruction
instruction = await learning.generate_personalized_instruction(
    student_id=student_id,
    target_concept="class_inheritance"
)
# AI will create visual, example-heavy instruction knowing student's learning style
```

**Benefits**:
- Personalized learning paths based on individual progress
- Early identification of knowledge gaps
- Adaptive instruction matching learning styles
- Long-term tracking of concept mastery

## Healthcare and Wellness

### Personal Health Assistant

**Challenge**: Health applications need to track symptoms, treatments, lifestyle factors, and provide contextual health guidance while maintaining privacy.

**Solution**: Secure memory system tracks health patterns, medication effectiveness, and lifestyle correlations.

```python
class HealthAssistant:
    def __init__(self):
        self.client = MemoryAPIClient(base_url="http://localhost:8000")
        self.namespace = "health_private"

    async def track_symptom(self, user_id: str, symptom: str, severity: int,
                          triggers: list = None, context: str = None):
        """Track symptom occurrence with context"""
        memory_text = f"Symptom: {symptom}, severity {severity}/10"
        if triggers:
            memory_text += f", potential triggers: {', '.join(triggers)}"
        if context:
            memory_text += f", context: {context}"

        await self.client.create_long_term_memories([{
            "text": memory_text,
            "memory_type": "episodic",
            "event_date": datetime.now().isoformat(),
            "topics": ["symptoms", symptom] + (triggers or []),
            "entities": [symptom, f"severity_{severity}"] + (triggers or []),
            "user_id": user_id,
            "namespace": self.namespace
        }])

    async def track_treatment_effectiveness(self, user_id: str, treatment: str,
                                         effectiveness: str, side_effects: list = None):
        """Track treatment outcomes"""
        memory_text = f"Treatment {treatment}: {effectiveness} effectiveness"
        if side_effects:
            memory_text += f", side effects: {', '.join(side_effects)}"

        await self.client.create_long_term_memories([{
            "text": memory_text,
            "memory_type": "episodic",
            "event_date": datetime.now().isoformat(),
            "topics": ["treatment", "effectiveness", treatment],
            "entities": [treatment, effectiveness] + (side_effects or []),
            "user_id": user_id,
            "namespace": self.namespace
        }])

    async def identify_patterns(self, user_id: str, focus_area: str):
        """Identify health patterns over time"""
        pattern_search = await self.client.search_long_term_memory(
            text=f"{focus_area} symptoms patterns triggers",
            filters={
                "user_id": {"eq": user_id},
                "namespace": {"eq": self.namespace}
            },
            limit=20,
            recency_boost=True
        )

        return pattern_search

    async def get_contextual_health_guidance(self, user_id: str, current_concern: str):
        """Provide personalized health guidance"""
        guidance_prompt = await self.client.memory_prompt(
            query=f"Provide guidance for: {current_concern}",
            long_term_search={
                "text": f"{current_concern} symptoms treatments patterns",
                "filters": {
                    "user_id": {"eq": user_id},
                    "namespace": {"eq": self.namespace}
                },
                "limit": 8
            }
        )

        return guidance_prompt

# Usage example (with appropriate privacy safeguards)
health = HealthAssistant()
user_id = "user_private_health_id"

# Track symptoms with context
await health.track_symptom(
    user_id=user_id,
    symptom="headache",
    severity=6,
    triggers=["stress", "screen_time"],
    context="end of work week, long computer sessions"
)

# Track treatment effectiveness
await health.track_treatment_effectiveness(
    user_id=user_id,
    treatment="reduced_screen_time",
    effectiveness="moderate_improvement",
    side_effects=["initial_productivity_concerns"]
)

# Identify patterns
patterns = await health.identify_patterns(user_id, "headache")

# Get contextual guidance
guidance = await health.get_contextual_health_guidance(
    user_id=user_id,
    current_concern="recurring headaches during work"
)
# AI will have context about screen time correlation and previous treatment success
```

**Benefits**:
- Long-term health pattern recognition
- Personalized treatment tracking and optimization
- Trigger identification and avoidance strategies
- Context-aware health guidance

## Best Practices Across Use Cases

### Memory Organization
- **Use namespaces**: Organize memories by domain, project, or context
- **Consistent tagging**: Use standardized topics and entities for better search
- **Appropriate memory types**: Semantic for facts, episodic for events

### Search Optimization
- **Enable recency boost**: For time-sensitive domains like support or health
- **Use query optimization**: For natural language queries from end users
- **Filter strategically**: Combine semantic search with metadata filters

### Privacy and Security
- **User isolation**: Always filter by user_id for personal data
- **Namespace separation**: Isolate sensitive domains (health, finance)
- **Authentication**: Enable appropriate auth for production deployments

### Performance
- **Batch operations**: Use bulk memory creation for initial data loading
- **Background processing**: Let automatic promotion handle memory management
- **Regular cleanup**: Use forgetting mechanisms for outdated information

These use cases demonstrate the versatility of Redis Agent Memory Server across industries and applications. The key is to design your memory schema and search patterns to match your specific domain needs while leveraging the platform's intelligent features for optimal user experiences.
