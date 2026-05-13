import os
from pathlib import Path
from datetime import datetime

videos_dir = Path(r"c:\Users\manny\Documents\BUS696\Chapman_FSM_DCF\Videos\BUS_696_LBO")
output_file = videos_dir / "LBO_TRANSCRIPTS_SUMMARY.md"

# Find all transcript files
transcripts = sorted(videos_dir.rglob("*.transcript.txt"))

# Start building the markdown
md_content = []
md_content.append("# Leveraged Buyout (LBO) Modeling Video Transcripts - Complete Summary\n")
md_content.append(f"**Date Generated:** {datetime.now().strftime('%B %d, %Y')}\n")
md_content.append(f"**Total Transcripts:** {len(transcripts)}\n")
md_content.append("**Location:** `Chapman_FSM_DCF/Videos/BUS_696_LBO/`\n\n")
md_content.append("---\n\n")

# Table of Contents
md_content.append("## Table of Contents\n\n")
for i, transcript in enumerate(transcripts, 1):
    topic_name = transcript.parent.parent.name
    anchor = topic_name.lower().replace(" ", "-").replace("--", "-")
    md_content.append(f"{i}. [{topic_name}](#{anchor})\n")

md_content.append("\n---\n\n")

# Navigation Instructions
md_content.append("## Navigation Instructions for AI\n\n")
md_content.append("Each transcript is located at:\n\n")
md_content.append("```\n")
md_content.append("Chapman_FSM_DCF/Videos/BUS_696_LBO/{TopicName}/transcript/{TopicName}.transcript.txt\n")
md_content.append("```\n\n")
md_content.append("**Full Windows Path Pattern:**\n")
md_content.append("```\n")
md_content.append("c:\\Users\\manny\\Documents\\BUS696\\Chapman_FSM_DCF\\Videos\\BUS_696_LBO\\{TopicName}\\transcript\\{TopicName}.transcript.txt\n")
md_content.append("```\n\n")
md_content.append("---\n\n")

# Add all transcripts
md_content.append("## Complete Transcripts\n\n")

count = 0
for transcript in transcripts:
    # Check if file exists before processing
    if not transcript.exists():
        print(f"Skipping missing file: {transcript}")
        continue
    
    count += 1
    topic_name = transcript.parent.parent.name
    anchor = topic_name.lower().replace(" ", "-").replace("--", "-")
    
    md_content.append(f"### {count}. {topic_name}\n\n")
    md_content.append(f"**Relative Path:** `Chapman_FSM_DCF/Videos/BUS_696_LBO/{topic_name}/transcript/{topic_name}.transcript.txt`\n\n")
    md_content.append(f"**Full Path:** `{transcript}`\n\n")
    md_content.append("---\n\n")
    
    # Read transcript content
    try:
        with open(transcript, 'r', encoding='utf-8') as f:
            content = f.read()
        md_content.append(content)
        md_content.append("\n\n")
        md_content.append("---\n\n")
    except Exception as e:
        print(f"Error reading {transcript}: {e}")
        md_content.append(f"*Error reading file: {e}*\n\n")
        md_content.append("---\n\n")

# Write to file
with open(output_file, 'w', encoding='utf-8') as f:
    f.writelines(md_content)

print(f"✓ Markdown file created successfully!")
print(f"✓ Location: {output_file}")
print(f"✓ Total transcripts: {count}")
print(f"✓ File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
