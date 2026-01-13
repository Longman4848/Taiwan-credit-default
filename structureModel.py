import os

print("📋 Taiwan Credit Default Model Checker")
print("=" * 50)

# Check if data folder exists
if os.path.exists("data"):
    print("✅ data/ folder exists")
    
    # Check if mlruns exists
    if os.path.exists("data/mlruns"):
        print("✅ data/mlruns/ folder exists")
        
        # Check experiment 2
        if os.path.exists("data/mlruns/2"):
            print("✅ data/mlruns/2/ folder exists")
            
            # Check your specific run
            run_path = "data/mlruns/2/c62fb037df3248fb9ef29f8f88d35a3e"
            if os.path.exists(run_path):
                print("✅ Your run folder exists!")
                
                # Check artifacts
                artifacts_path = os.path.join(run_path, "artifacts")
                if os.path.exists(artifacts_path):
                    print("✅ artifacts/ folder exists")
                    
                    # Check xgboost_model
                    model_path = os.path.join(artifacts_path, "xgboost_model")
                    if os.path.exists(model_path):
                        print("✅ xgboost_model/ folder exists!")
                        print()
                        print("🎯 FOUND YOUR MODEL!")
                        print(f"📋 Use this path in FastAPI:")
                        print(f"   model_path = 'data/mlruns/2/c62fb037df3248fb9ef29f8f88d35a3e/artifacts/xgboost_model'")
                        
                        # List model contents
                        print("\n📁 Model contents:")
                        for item in os.listdir(model_path):
                            item_path = os.path.join(model_path, item)
                            if os.path.isfile(item_path):
                                size = os.path.getsize(item_path)
                                print(f"   📄 {item} ({size:,} bytes)")
                            else:
                                print(f"   📁 {item}/")
                    else:
                        print("❌ xgboost_model/ folder NOT found")
                        print("   Available in artifacts:")
                        for item in os.listdir(artifacts_path):
                            print(f"   - {item}")
                else:
                    print("❌ artifacts/ folder NOT found")
            else:
                print("❌ Your run folder NOT found")
                print("   Available in mlruns/2/:")
                for item in os.listdir("data/mlruns/2/"):
                    print(f"   - {item}")
        else:
            print("❌ data/mlruns/2/ folder NOT found")
            print("   Available in data/mlruns/:")
            for item in os.listdir("data/mlruns"):
                print(f"   - {item}")
    else:
        print("❌ data/mlruns/ folder NOT found")
        print("   Available in data/:")
        for item in os.listdir("data"):
            print(f"   - {item}")
else:
    print("❌ data/ folder NOT found")
    print("   Available folders:")
    for item in os.listdir("."):
        if os.path.isdir(item):
            print(f"   - {item}")
    print("   Available files:")
    for item in os.listdir("."):
        if os.path.isfile(item):
            print(f"   - {item}")

print("\n" + "=" * 50)
print("💡 Make sure you're running this from the project root!")