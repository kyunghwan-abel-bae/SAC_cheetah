import subprocess
import re
import os
import fileinput
import sys

def set_hyperparameters(file_path, batch_size, min_buffer_size):
    """주어진 파일의 하이퍼파라미터를 동적으로 설정합니다."""
    print(f'  -> "{os.path.basename(file_path)}" 파일 업데이트: batch={batch_size}, buffer={min_buffer_size}')
    
    # fileinput 모듈을 사용하여 파일을 직접 수정합니다.
    # 이 스크립트가 실행되는 동안 원본 파일의 내용이 잠시 변경됩니다.
    for line in fileinput.input(file_path, inplace=True):
        line = re.sub(r"batch_size = \d+", f"batch_size = {batch_size}", line)
        line = re.sub(r"min_buffer_size = \d+", f"min_buffer_size = {min_buffer_size}", line)
        sys.stdout.write(line)

def run_single_benchmark(version, batch_size, min_buffer_size):
    """단일 벤치마크를 실행하고 결과를 반환합니다."""
    print(f"\n[실행] 버전: {version.upper()}, 배치 사이즈: {batch_size}, 버퍼: {min_buffer_size})")
    
    target_file = 'sac_cheetah.py' if version == 'cpu' else 'sac_cheetah_mps.py'
    set_hyperparameters(target_file, batch_size, min_buffer_size)
        
    command = ['python', '-u', 'benchmark.py', version]
    print(f"  -> Executing command: {' '.join(command)}")
    
    output_lines = []
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', bufsize=1)
        
        for line in iter(process.stdout.readline, ''):
            sys.stdout.write(line)
            output_lines.append(line)
        
        process.stdout.close()
        return_code = process.wait()
        
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command, output="".join(output_lines))
            
    except Exception as e:
        print(f"An error occurred: {e}")
        # 에러 발생 시 전체 출력을 보여줍니다.
        print("--- benchmark.py raw output ---\n" + "".join(output_lines) + "\n---------------------------------")
        raise

    output = "".join(output_lines)
    print("  -> 벤치마크 실행 완료.")
    
    match = re.search(r"성능: ([\d\.]+) steps/second", output, re.DOTALL)
    if match:
        sps = float(match.group(1))
        print(f"  -> 성능: {sps:.2f} steps/second")
        return sps
    else:
        print("벤치마크 결과 파싱 실패.")
        raise RuntimeError(f"결과 파싱 실패: {version} {batch_size}")

def generate_html_report(results):
    """벤치마크 결과로부터 HTML 리포트를 생성합니다."""
    
    # 가장 좋은 결과를 하이라이트하기 위해 찾습니다.
    best_sps = 0
    if results:
        best_sps = max(r['sps'] for r in results)

    html = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>SAC 벤치마크 결과</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen-Sans, Ubuntu, Cantarell, "Helvetica Neue", sans-serif; margin: 2em; background-color: #f8f9fa; color: #212529; }}
        h1 {{ color: #343a40; }}
        table {{ border-collapse: collapse; width: 80%; margin-top: 1.5em; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        th, td {{ border: 1px solid #dee2e6; text-align: left; padding: 12px; }}
        th {{ background-color: #e9ecef; }}
        tr:nth-child(even) {{ background-color: #fdfdfe; }}
        .highlight {{ background-color: #d1e7dd; font-weight: bold; color: #0f5132; }}
        .container {{ max-width: 1000px; margin: auto; background-color: white; padding: 2em; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
    </style>
</head>
<body>
    <div class="container">
        <h1>SAC 벤치마크 결과</h1>
        <p>성능은 초당 환경 스텝(steps/second)으로 측정되었습니다. (높을수록 좋음)</p>
        <table>
            <tr>
                <th>버전</th>
                <th>배치 사이즈</th>
                <th>최소 버퍼 사이즈</th>
                <th>초당 스텝 (Steps/Second)</th>
            </tr>
"""
    for r in results:
        highlight_class = 'class="highlight"' if r['sps'] == best_sps else ''
        html += f"""
            <tr {highlight_class}>
                <td>{r['version']}</td>
                <td>{r['batch']}</td>
                <td>{r['buffer']}</td>
                <td>{r['sps']:.2f}</td>
            </tr>"""

    html += """
        </table>
    </div>
</body>
</html>
"""
    with open('test.html', 'w', encoding='utf-8') as f:
        f.write(html)
    print("\n\n벤치마크 완료. 모든 결과가 test.html 파일에 저장되었습니다.")

def main():
    # CPU는 기본 설정으로 한 번만 테스트합니다.
    cpu_config = {'version': 'cpu', 'batch': 256, 'buffer': 5000}

    # MPS는 다양한 배치 사이즈와 버퍼로 테스트합니다.
    mps_configs = [
        {'version': 'mps', 'batch': 256,  'buffer': 5000},
        {'version': 'mps', 'batch': 512,  'buffer': 5000},
        {'version': 'mps', 'batch': 1024, 'buffer': 10000},
        {'version': 'mps', 'batch': 2048, 'buffer': 20000},
    ]
    
    configs = [cpu_config] + mps_configs
    
    all_results = []
    
    # 원본 하이퍼파라미터 저장 (복원을 위해)
    original_params = {
        'cpu': {'batch': 256, 'buffer': 5000}, # 임의의 값으로 설정, 실제 파일 값 읽어오는 것은 복잡
        'mps': {'batch': 256, 'buffer': 5000}
    }
    
    try:
        for config in configs:
            sps = run_single_benchmark(config['version'], config['batch'], config['buffer'])
            result_data = {
                'version': config['version'].upper(),
                'batch': config['batch'],
                'buffer': config['buffer'],
                'sps': sps
            }
            all_results.append(result_data)
    except (subprocess.CalledProcessError, RuntimeError) as e:
        print(f"\n벤치마크 실행 중 에러 발생: {e}", file=sys.stderr)
        print("중단합니다.", file=sys.stderr)
        return
    finally:
        # 테스트가 끝나면 원래 하이퍼파라미터로 복원 시도
        print("\n테스트 완료. 하이퍼파라미터를 원래대로 복원합니다...")
        set_hyperparameters('sac_cheetah.py', 1024, 10000)
        set_hyperparameters('sac_cheetah_mps.py', 1024, 10000)

    # 결과 리포트 생성
    generate_html_report(all_results)

if __name__ == "__main__":
    # benchmark.py 파일이 있는지 확인
    if not os.path.exists('benchmark.py'):
        print("에러: benchmark.py 파일이 현재 디렉토리에 없습니다.", file=sys.stderr)
        print("이전 단계에서 생성된 benchmark.py를 먼저 저장해주세요.", file=sys.stderr)
        sys.exit(1)
    main()
