#!/usr/bin/env python3
"""
DreamLog Background Learning Service Demo

This demo shows how to use the background learning service for persistent
learning with inter-process communication. It demonstrates:

1. Starting the background learning service
2. Using the client API to interact with the service
3. Monitoring service status and metrics
4. Managing knowledge injection and querying
5. Controlling sleep cycles remotely

This is useful for production deployments where you want a long-running
learning service that multiple clients can interact with.
"""

import time
import tempfile
import shutil
import threading
from pathlib import Path

from dreamlog.background_learner import BackgroundLearner, BackgroundLearnerClient
from dreamlog.sleep_cycle import SleepPhase
from dreamlog.knowledge import Fact, Rule
from dreamlog.factories import atom, var, compound
from dreamlog.logging_config import setup_logging, get_logger

# Setup logging
setup_logging(enable_file_logging=False, log_level="INFO")
logger = get_logger(__name__)


def demo_background_service():
    """Demo the background learning service"""
    print("\n" + "="*60)
    print("DreamLog Background Learning Service Demo")
    print("="*60)
    
    # Create temporary storage
    temp_dir = Path(tempfile.mkdtemp())
    service_port = 7781  # Use unique port to avoid conflicts
    
    print(f"Using temporary storage: {temp_dir}")
    print(f"Service port: {service_port}")
    
    # Initialize background service
    print("\n1. Starting background learning service...")
    service = BackgroundLearner(
        storage_path=temp_dir,
        llm_provider=None,  # No LLM for this demo
        ipc_port=service_port,
        config={
            "sleep_config": {
                "light_sleep_interval": {"seconds": 30},  # Fast for demo
                "min_idle_time": {"seconds": 2}
            }
        }
    )
    
    try:
        # Start service in background
        service.start()
        time.sleep(2)  # Give service time to start up
        
        print("   Service started successfully")
        
        # Create client
        print("\n2. Creating client connection...")
        client = BackgroundLearnerClient(service_port)
        
        # Test basic connectivity
        print("\n3. Testing service connectivity...")
        status = client.get_status()
        print(f"   Service status: {status['status']}")
        print(f"   Service uptime: {status['uptime_seconds']:.1f} seconds")
        
        # Add some knowledge
        print("\n4. Adding knowledge via client...")
        facts = [
            Fact(compound("parent", atom("john"), atom("mary"))),
            Fact(compound("parent", atom("mary"), atom("alice"))),
            Fact(compound("parent", atom("bob"), atom("tom"))),
            Fact(compound("likes", atom("john"), atom("pizza")))
        ]
        
        rules = [
            Rule(
                compound("grandparent", var("X"), var("Z")),
                [compound("parent", var("X"), var("Y")), compound("parent", var("Y"), var("Z"))]
            ),
            Rule(
                compound("ancestor", var("X"), var("Y")),
                [compound("parent", var("X"), var("Y"))]
            )
        ]
        
        result = client.add_user_knowledge(facts, rules)
        print(f"   Facts added: {result['facts_added']}")
        print(f"   Rules added: {result['rules_added']}")
        print(f"   Conflicts detected: {result['conflicts_detected']}")
        print(f"   Conflicts resolved: {result['conflicts_resolved']}")
        
        # Query knowledge
        print("\n5. Querying knowledge...")
        queries = [
            [compound("parent", atom("john"), var("X"))],
            [compound("grandparent", var("X"), var("Y"))],
            [compound("likes", var("X"), atom("pizza"))]
        ]
        
        for i, query in enumerate(queries, 1):
            solutions = client.query(query)
            query_str = str(query[0]) if query else "unknown"
            print(f"   Query {i} '{query_str}': {len(solutions)} solutions")
            
            for j, solution in enumerate(solutions[:2], 1):  # Show first 2
                print(f"     {j}. {solution}")
        
        # Monitor service status
        print("\n6. Service status after operations...")
        status = client.get_status()
        if 'current_session' in status and status['current_session']:
            session = status['current_session']
            print(f"   Queries processed: {session.get('queries_processed', 0)}")
            print(f"   Facts learned: {session.get('facts_learned', 0)}")
            print(f"   Rules learned: {session.get('rules_learned', 0)}")
        
        # Check KB metrics
        if 'kb_metrics' in status:
            kb = status['kb_metrics']
            if 'kb1_size' in kb:
                print(f"   KB_1 size: {kb['kb1_size']['facts']} facts, {kb['kb1_size']['rules']} rules")
            if 'kb2_size' in kb:
                print(f"   KB_2 size: {kb['kb2_size']['facts']} facts, {kb['kb2_size']['rules']} rules")
        
        # Force a sleep cycle
        print("\n7. Forcing sleep cycle...")
        sleep_result = client.force_sleep_cycle(SleepPhase.LIGHT_SLEEP)
        print(f"   Cycle ID: {sleep_result['cycle_id']}")
        print(f"   Phase: {sleep_result['phase']}")
        print(f"   Operations: {sleep_result['operations']}")
        print(f"   Duration: {sleep_result['duration_ms']:.1f}ms")
        print(f"   Compression ratio: {sleep_result['compression_ratio']:.2%}")
        
        # Final status check
        print("\n8. Final service status...")
        final_status = client.get_status()
        
        if 'sleep_metrics' in final_status:
            sleep_metrics = final_status['sleep_metrics']
            if 'total_cycles' in sleep_metrics:
                print(f"   Total sleep cycles: {sleep_metrics['total_cycles']}")
                print(f"   Facts removed: {sleep_metrics.get('total_facts_removed', 0)}")
                print(f"   Rules generated: {sleep_metrics.get('total_rules_generated', 0)}")
        
        # Test performance
        print("\n9. Performance test...")
        start_time = time.time()
        
        # Run multiple queries
        for _ in range(10):
            solutions = client.query([compound("parent", var("X"), var("Y"))])
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 10 * 1000  # Convert to ms
        print(f"   Average query time: {avg_time:.1f}ms")
        
        print("\n10. Shutting down service...")
        client.shutdown()
        time.sleep(1)  # Give service time to shut down gracefully
        
        print("   Service shutdown completed")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Ensure service is stopped
        try:
            service.stop()
        except:
            pass
        
        # Clean up storage
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temporary storage")


def demo_multiple_clients():
    """Demo multiple clients connecting to the same service"""
    print("\n" + "="*60)
    print("Multiple Clients Demo")
    print("="*60)
    
    temp_dir = Path(tempfile.mkdtemp())
    service_port = 7782
    
    print(f"Starting service on port {service_port}...")
    
    service = BackgroundLearner(
        storage_path=temp_dir,
        ipc_port=service_port
    )
    
    try:
        service.start()
        time.sleep(2)
        
        print("Service started successfully")
        
        # Function for client operations
        def client_operations(client_id: int, num_operations: int):
            client = BackgroundLearnerClient(service_port)
            
            for i in range(num_operations):
                # Add knowledge
                fact = Fact(compound("data", atom(f"client_{client_id}"), atom(f"item_{i}")))
                client.add_user_knowledge([fact], [])
                
                # Query knowledge
                solutions = client.query([compound("data", var("X"), var("Y"))])
                
                print(f"Client {client_id}, Op {i+1}: Added fact, found {len(solutions)} total items")
                
                time.sleep(0.1)  # Small delay
        
        print("\nStarting multiple clients...")
        
        # Start 3 clients concurrently
        threads = []
        for client_id in range(3):
            thread = threading.Thread(
                target=client_operations,
                args=(client_id, 5)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all clients to complete
        for thread in threads:
            thread.join()
        
        # Check final state
        print("\nFinal state check...")
        client = BackgroundLearnerClient(service_port)
        status = client.get_status()
        
        if 'kb_metrics' in status:
            kb = status['kb_metrics']
            total_facts = kb.get('kb1_size', {}).get('facts', 0) + kb.get('kb2_size', {}).get('facts', 0)
            print(f"Total facts in KB: {total_facts}")
        
        # Query all data items
        solutions = client.query([compound("data", var("X"), var("Y"))])
        print(f"Total data items: {len(solutions)}")
        
        print("\nMultiple clients demo completed successfully")
        
    except Exception as e:
        print(f"Multiple clients demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        try:
            service.stop()
        except:
            pass
        shutil.rmtree(temp_dir)


def demo_service_persistence():
    """Demo service persistence across restarts"""
    print("\n" + "="*60)
    print("Service Persistence Demo")
    print("="*60)
    
    temp_dir = Path(tempfile.mkdtemp())
    service_port = 7783
    
    print("Phase 1: Initial service with knowledge...")
    
    # First service instance
    service1 = BackgroundLearner(
        storage_path=temp_dir,
        ipc_port=service_port
    )
    
    try:
        service1.start()
        time.sleep(2)
        
        client = BackgroundLearnerClient(service_port)
        
        # Add initial knowledge
        facts = [
            Fact(compound("persistent", atom("fact1"))),
            Fact(compound("persistent", atom("fact2"))),
            Fact(compound("temp", atom("data")))
        ]
        
        result = client.add_user_knowledge(facts, [])
        print(f"Added {result['facts_added']} facts to first service instance")
        
        # Query to verify
        solutions = client.query([compound("persistent", var("X"))])
        print(f"Found {len(solutions)} persistent facts")
        
        # Shutdown first service
        print("\nShutting down first service instance...")
        client.shutdown()
        time.sleep(2)
        
    except Exception as e:
        print(f"Phase 1 failed: {e}")
        return
    finally:
        try:
            service1.stop()
        except:
            pass
    
    print("\nPhase 2: Restarting service with same storage...")
    
    # Second service instance (same storage)
    service2 = BackgroundLearner(
        storage_path=temp_dir,
        ipc_port=service_port
    )
    
    try:
        service2.start()
        time.sleep(2)
        
        client = BackgroundLearnerClient(service_port)
        
        # Check if knowledge persisted
        solutions = client.query([compound("persistent", var("X"))])
        print(f"After restart, found {len(solutions)} persistent facts")
        
        if len(solutions) > 0:
            print("✓ Knowledge successfully persisted across service restart!")
            for i, solution in enumerate(solutions, 1):
                print(f"  {i}. {solution}")
        else:
            print("✗ Knowledge was not persisted")
        
        # Add more knowledge
        new_facts = [Fact(compound("after_restart", atom("new_data")))]
        result = client.add_user_knowledge(new_facts, [])
        print(f"\nAdded {result['facts_added']} new facts after restart")
        
        # Verify total
        all_solutions = client.query([compound(var("P"), var("X"))])
        print(f"Total facts in restarted service: {len(all_solutions)}")
        
        print("\nService persistence demo completed successfully")
        
    except Exception as e:
        print(f"Phase 2 failed: {e}")
    
    finally:
        try:
            service2.stop()
        except:
            pass
        shutil.rmtree(temp_dir)


def run_all_background_demos():
    """Run all background service demos"""
    print("DreamLog Background Learning Service Demos")
    print("="*60)
    print("These demos showcase the background learning service that enables")
    print("persistent learning with inter-process communication.")
    print()
    
    try:
        demo_background_service()
        demo_multiple_clients()
        demo_service_persistence()
        
        print("\n" + "="*60)
        print("ALL BACKGROUND SERVICE DEMOS COMPLETED!")
        print("="*60)
        print()
        print("The background learning service provides:")
        print("✓ Long-running persistent learning process")
        print("✓ Inter-process communication via TCP sockets")
        print("✓ Multiple client support")
        print("✓ Knowledge persistence across restarts")
        print("✓ Remote sleep cycle control")
        print("✓ Real-time status and metrics monitoring")
        print()
        print("This enables production deployments where multiple")
        print("applications can share a single learning service.")
        
    except Exception as e:
        print(f"\nBackground service demos failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_background_demos()