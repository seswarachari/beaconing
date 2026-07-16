import subprocess
import sys
import os

def run_command(command, description):
    print(f"\n{'='*60}")
    print(f"🚀 STARTING: {description}")
    print(f"Executing: {command}")
    print(f"{'='*60}\n")
    
    try:
        # Run the command and stream the output to the console
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=sys.stdout, 
            stderr=sys.stderr
        )
        process.communicate()
        
        if process.returncode != 0:
            print(f"\n❌ ERROR: {description} failed with exit code {process.returncode}")
            sys.exit(process.returncode)
            
        print(f"\n✅ SUCCESS: {description} completed successfully!\n")
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR running {description}: {e}")
        sys.exit(1)

def main():
    # Ensure we are in the correct directory (the root of the project)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("Welcome to the C2 Beaconing Detection Engine!")
    print("This script will launch the Streamlit Dashboard.\n")
    
    # Launch the Streamlit Dashboard
    print(f"\n{'='*60}")
    print("🚀 STARTING: Launching the Interactive Dashboard")
    print("The dashboard will open in your default web browser.")
    print("Please use the 'Upload Data' sidebar to analyze your PCAP files.")
    print("Press Ctrl+C in this terminal to stop the server when you are done.")
    print(f"{'='*60}\n")
    
    try:
        # Use subprocess.run for streamlit since it's a long-running process
        subprocess.run(f"streamlit run dashboard/app.py", shell=True)
    except KeyboardInterrupt:
        print("\n\n🛑 Dashboard stopped by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error launching dashboard: {e}")

if __name__ == "__main__":
    main()
