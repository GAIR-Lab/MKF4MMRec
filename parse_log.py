import os
import re
import csv
import time
import shutil

# Parse metric string into a dict
def parse_metrics(line):
    metrics = {}
    parts = re.findall(r'(\w+@\d+):\s*([\d\.]+)', line)
    for key, value in parts:
        metrics[key] = float(value)
    return metrics

# Parse a single log file
def parse_log_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Search from the end of the file, find the last BEST flag, then find Valid and Test
    best_line = -1
    for i in range(len(lines) - 1, -1, -1):
        if '█████████████ BEST ████████████████' in lines[i]:
            best_line = i
            break
    if best_line == -1:
        return None  # No BEST flag, skip

    valid_line, test_line = None, None
    for i in range(best_line, len(lines)):
        if 'Valid:' in lines[i]:
            valid_line = lines[i]
        if 'Test:' in lines[i]:
            test_line = lines[i]

    if not valid_line or not test_line:
        return None  # Missing valid/test line

    return {
        'valid': parse_metrics(valid_line),
        'test': parse_metrics(test_line)
    }

def move_incomplete_logs(log_dir, temp_dir='temp_log'):
    """
    Check and move incomplete log files to the temp_log folder.
    If a file does not have a BEST flag and its last modification time is more than 30 minutes ago, it is considered an incomplete training.
    """
    # Ensure the temp_log folder exists
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    current_time = time.time()
    moved_files = []
    
    for filename in os.listdir(log_dir):
        if filename.endswith('.log'):
            file_path = os.path.join(log_dir, filename)
            
            # Check the last modification time of the file
            last_modified = os.path.getmtime(file_path)
            time_diff = current_time - last_modified
            
            # If the file modification time is more than 30 minutes (1800 seconds)
            if time_diff > 1800:
                # Check for BEST flag
                has_best = False
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        if '█████████████ BEST ████████████████' in content:
                            has_best = True
                except Exception as e:
                    print(f'[ERROR] Failed to read {filename}: {e}')
                    continue
                
                # If there is no BEST flag, move it to temp_log
                if not has_best:
                    temp_file_path = os.path.join(temp_dir, filename)
                    try:
                        shutil.move(file_path, temp_file_path)
                        moved_files.append(filename)
                        print(f'[MOVED] {filename} -> {temp_dir}/ (incomplete training)')
                    except Exception as e:
                        print(f'[ERROR] Failed to move {filename}: {e}')
    
    return moved_files

# Parse all log files
def parse_all_logs(log_dir):
    results = []
    for filename in os.listdir(log_dir):
        if filename.endswith('.log'):
            file_path = os.path.join(log_dir, filename)
            parsed = parse_log_file(file_path)
            if parsed:
                results.append((filename.replace('.log', ''), parsed))
            else:
                print(f'[WARN] Skipped (incomplete): {filename}')
    return results

# Save to CSV
def sort_metrics(metrics):
    """
    Group metrics by type and sort by the number after @
    e.g., map@5, map@10, recall@5, recall@10
    """
    def metric_key(m):
        match = re.match(r'(\D+?)@(\d+)', m)
        if match:
            name, k = match.groups()
            return (name, int(k))
        return (m, 0)
    
    return sorted(metrics, key=metric_key)

def save_to_csv(results, output_file='log_summary.csv'):
    # Collect all occurring metrics
    all_metrics = set()
    for _, result in results:
        all_metrics.update(result['valid'].keys())
        all_metrics.update(result['test'].keys())

    # Sort metrics
    sorted_metrics = sort_metrics(all_metrics)

    # Build headers
    headers = ['model'] + [f'valid_{m}' for m in sorted_metrics] + [f'test_{m}' for m in sorted_metrics]

    # Sort by model name
    results.sort(key=lambda x: x[0].lower())

    # Write to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for model_name, result in results:
            row = [model_name]
            for m in sorted_metrics:
                row.append(result['valid'].get(m, ''))
            for m in sorted_metrics:
                row.append(result['test'].get(m, ''))
            writer.writerow(row)

# Main entry point
if __name__ == '__main__':
    log_dir = './log'  # Your log directory path
    
    # First, move incomplete log files
    print("Checking for incomplete log files...")
    moved_files = move_incomplete_logs(log_dir)
    if moved_files:
        print(f"Moved {len(moved_files)} incomplete log files to temp_log/")
    else:
        print("No incomplete log files found.")
    
    # Then parse the remaining log files
    print("\nParsing remaining log files...")
    results = parse_all_logs(log_dir)
    save_to_csv(results)
    print(f'Done. Parsed {len(results)} logs.')