#!/usr/bin/env python3
"""Fail-closed integration of the pinned pre-AST patch and source overlay.

The cumulative patch deliberately stops before the few upstream-owned methods
that need structural rewrites.  Bootstrap applies that patch, copies the
overlay, then invokes this program.  The overlay owns the calibrated tools;
this program must therefore never add a second calibration call to them.

No CPU, CUDA, or training validation is performed here.  Those checks are
intentionally NOT RUN for the current repair unless the user requests them.
"""
from __future__ import annotations
import argparse,ast,json,os
from pathlib import Path


def replace_methods(source):
    tree=ast.parse(source);lines=source.splitlines(keepends=True);edits=[]
    classes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GaussianModel']
    if len(classes)!=1:raise RuntimeError('Expected exactly one GaussianModel class')
    cls=classes[0]
    for n in cls.body:
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in ('capture','restore'):
            segment=ast.get_source_segment(source,n) or ''
            if 'diffusion_guidance.checkpoint_state' in segment:
                raise RuntimeError('Checkpoint delegation already present; use a fresh pre-AST checkout')
            indent=' '*n.col_offset; body_indent=' '*(n.col_offset+4)
            if n.name=='capture': body=f'{indent}def capture(self):\n{body_indent}from diffusion_guidance.checkpoint_state import capture\n{body_indent}return capture(self)\n'
            else: body=f'{indent}def restore(self, model_args, training_args):\n{body_indent}from diffusion_guidance.checkpoint_state import restore\n{body_indent}return restore(self, model_args, training_args)\n'
            edits.append((n.lineno-1,n.end_lineno,body))
    if len(edits)!=2:raise RuntimeError('Unexpected GaussianModel methods')
    for start,end,body in sorted(edits,reverse=True):lines[start:end]=[body]
    result=''.join(lines);ast.parse(result);return result


def _replace_at_most_once(source, old, new, label):
    """Keep helper-level AST fixtures small while rejecting ambiguous rewrites."""
    count=source.count(old)
    if count>1:raise RuntimeError(f'Unexpected {label} count: {count}')
    return source.replace(old,new,1) if count else source


def fix_training(source):
    tree=ast.parse(source);lines=source.splitlines(keepends=True)
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='training']
    if len(functions)!=1:raise RuntimeError('Expected exactly one top-level training() function')
    fn=functions[0]
    blocks=[];reset=None;reports=[]
    for n in ast.walk(fn):
        if isinstance(n,ast.If):
            segment=ast.get_source_segment(source,n) or ''
            if 'scene.save(iteration)' in segment and 'saving_iterations' in ast.unparse(n.test):blocks.append(n)
            if 'gaussians.capture()' in segment and 'checkpoint_iterations' in ast.unparse(n.test):blocks.append(n)
        if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call):
            if ast.unparse(n.value.func)=='gaussians.reset_opacity':reset=n
            if ast.unparse(n.value.func)=='training_report':reports.append(n)
    if len(blocks)!=2 or reset is None:raise RuntimeError('Unexpected save/reset structure; refusing unsafe rewrite')
    saved='\n'+''.join(''.join(lines[n.lineno-1:n.end_lineno])+'\n' for n in reports)+''.join(''.join(lines[n.lineno-1:n.end_lineno])+'\n' for n in sorted(blocks,key=lambda n:n.lineno))
    # Insert checkpoints after the optimizer update and any reset at this iteration.
    edits=[(n.lineno-1,n.end_lineno,'') for n in blocks+reports]+[(reset.end_lineno,reset.end_lineno,saved)]
    for start,end,text in sorted(edits,reverse=True):lines[start:end]=[text]
    result=''.join(lines)
    expected='if iteration < opt.iterations:'
    if result.count(expected)!=1:raise RuntimeError('Unexpected optimizer end condition')
    result=result.replace(expected,'if iteration <= opt.iterations:',1)
    result=_replace_at_most_once(result,'prepare_output_and_logger(dataset)',
                                 'prepare_output_and_logger(args)','output logger call')
    result=_replace_at_most_once(result,'torch.load(checkpoint)',
                                 'torch.load(checkpoint, weights_only=False)','checkpoint load call')
    # Legacy mono-depth must not download weights merely by importing train.py.
    result=_replace_at_most_once(result,'from utils.depth_utils import estimate_depth',
        'def estimate_depth(*args, **kwargs):\n    from utils.depth_utils import estimate_depth as implementation\n    return implementation(*args, **kwargs)','depth import')
    controlled_anchor='    first_iter = 0'
    if result.count(controlled_anchor)>1:
        raise RuntimeError('Unexpected first_iter initialization; refusing unsafe rewrite')
    if controlled_anchor in result:
        result=result.replace(controlled_anchor, '''    first_iter = 0
    if str(getattr(opt, "controlled_ab_role", "")).upper() in {"A0", "A1", "SELFRENDER", "B"}:
        # One identifiable source RGB + repaired track objective, no extra legacy module.
        args.geometry_reg_enabled = False
        opt.geometry_reg_enabled = False
        opt.disable_depth_loss = True
        args.disable_depth_loss = True
        if getattr(opt, "mixed_precision", False):
            raise RuntimeError("Controlled revision v2 requires full precision")
        if int(opt.densify_until_iter) != 10000:
            raise RuntimeError("Controlled v2 freezes topology at 10k")''',1)
    if 'calibrate_scene(' in result:
        raise RuntimeError('Training calibration already present; use a fresh pre-AST checkout')
    result=calibrate_scenes(result)
    sampler='        viewpoint_cam = viewpoint_stack.pop()'
    if sampler in result:
        if result.count('    viewpoint_stack, pseudo_stack = None, None') != 1:
            raise RuntimeError('Unexpected viewpoint stack initialization')
        if result.count(sampler) != 1:
            raise RuntimeError('Unexpected viewpoint selection count')
        result=result.replace('    viewpoint_stack, pseudo_stack = None, None',
                              '    real_view_audit = []\n    viewpoint_stack, pseudo_stack = None, None',1)
        result=result.replace(sampler,
                              sampler+'\n        real_view_audit.append([int(iteration), str(viewpoint_cam.image_name)])',1)
        tree=ast.parse(result);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='training')
        lines=result.splitlines(keepends=True)
        lines[fn.end_lineno:fn.end_lineno]=['\n    with open(os.path.join(scene.model_path, "real_view_sequence.json"), "w", encoding="utf-8") as handle:\n        json.dump(real_view_audit, handle)\n']
        result=''.join(lines)
    result='\n'.join(line.rstrip(' \t') for line in result.split('\n'))
    ast.parse(result);return result


def calibrate_scenes(source):
    if 'calibrate_scene(' in source:
        raise RuntimeError('Calibration already present; refusing duplicate insertion')
    tree=ast.parse(source);lines=source.splitlines(keepends=True);edits=[]
    for n in ast.walk(tree):
        if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='Scene':
            if len(n.targets)!=1 or not isinstance(n.targets[0],ast.Name):raise RuntimeError('Unexpected scene assignment')
            var=n.targets[0].id; indent=' '*n.col_offset
            addition=(f'{indent}if getattr(args, "strict_source_only_geometry", False):\n'
                       f'{indent}    from diffusion_guidance.calibration_guard import calibrate_scene\n'
                       f'{indent}    calibrate_scene({var}, args.track_path)\n')
            edits.append((n.end_lineno,addition))
    if not edits:raise RuntimeError('No Scene call to calibrate')
    for end,text in sorted(edits,reverse=True):lines[end:end]=[text]
    out=''.join(lines);ast.parse(out);return out


def validate_pre_ast_training(source):
    """The real package must have every anchor; unit fixtures need not."""
    tree=ast.parse(source)
    functions=[
        node for node in tree.body
        if isinstance(node,ast.FunctionDef) and node.name=='training'
    ]
    if len(functions)!=1:
        raise RuntimeError('Expected exactly one top-level training() function')
    training=functions[0]

    def calls(node, function_name):
        return any(
            isinstance(candidate,ast.Call)
            and ast.unparse(candidate.func)==function_name
            for candidate in ast.walk(node)
        )

    source_names_bound=any(
        isinstance(node,(ast.Assign,ast.AnnAssign))
        and any(
            isinstance(target,ast.Name) and target.id=='source_camera_names'
            for target in (
                node.targets if isinstance(node,ast.Assign) else [node.target]
            )
        )
        and calls(node.value,'normalized_camera_names')
        for node in ast.walk(training)
    )
    track_path_bound=any(
        isinstance(node,ast.Dict)
        and any(
            isinstance(key,ast.Constant)
            and key.value=='track_h5_path'
            and calls(value,'os.path.realpath')
            for key,value in zip(node.keys,node.values)
            if key is not None
        )
        for node in ast.walk(training)
    )
    required=(
        'if iteration < opt.iterations:',
        'prepare_output_and_logger(dataset)',
        'torch.load(checkpoint)',
        'from utils.depth_utils import estimate_depth',
        '    first_iter = 0',
        '    viewpoint_stack, pseudo_stack = None, None',
        '        viewpoint_cam = viewpoint_stack.pop()',
    )
    for anchor in required:
        if source.count(anchor)!=1:
            raise RuntimeError(f'Patch base has an unexpected integration anchor: {anchor}')
    for forbidden in ('calibrate_scene(', 'real_view_sequence.json',
                      'Controlled revision v2 requires full precision'):
        if forbidden in source:
            raise RuntimeError(f'Patch base already contains AST output: {forbidden}')
    # These anchors distinguish the final cumulative patch from earlier patches
    # that assembled successfully but emitted incomplete controlled-run evidence.
    required_research_contract=(
        'def _live_cuda_renderer_probe(',
        '"pseudo_rgb_start": int(opt.pseudo_rgb_start)',
        '"pseudo_rgb_end": int(opt.pseudo_rgb_end)',
        '"source_image_inventory_sha256": (',
        '"source_training_camera_inventory_sha256": (',
        '"controlled_training_protocol_sha256": (',
        '"start_checkpoint_controlled_provenance_sha256"',
        'source_cameras = list(scene.getTrainCameras()) if controlled_role else []',
        '_sha256_file(args.track_path) if controlled_role else None',
        'pseudo_records = load_pseudo_manifest(',
        'controlled_metadata["pseudo_manifest_sha256"]',
        'controlled_metadata["pseudo_camera_pool_sha256"]',
        'current_checkpoint_provenance = (',
        'gaussians.controlled_provenance = current_checkpoint_provenance',
        'Controlled {role} requires --pseudo_rgb_strict_cache',
        '"strict_geometry_protocol": {',
        '"densification_state_restored": bool(getattr(',
    )
    missing=[anchor for anchor in required_research_contract if anchor not in source]
    if not track_path_bound:
        missing.append('track_h5_path bound through os.path.realpath')
    if not source_names_bound:
        missing.append('source_camera_names bound through normalized_camera_names')
    if missing:
        raise RuntimeError(
            'Cumulative patch lacks the final controlled-run evidence contract: '
            f'{missing}'
        )
    pseudo_load=source.find('pseudo_records = load_pseudo_manifest(')
    provenance_build=source.find('current_checkpoint_provenance = (')
    provenance_assignment=source.find(
        'gaussians.controlled_provenance = current_checkpoint_provenance'
    )
    if not (0 <= pseudo_load < provenance_build < provenance_assignment):
        raise RuntimeError(
            'Cumulative patch must validate pseudo inputs before constructing and '
            'attaching final checkpoint provenance'
        )


def validate_overlay_contract(root):
    checkpoint=(root/'evidence_track/diffusion/checkpoint_state.py').read_text(encoding='utf-8')
    for anchor in (
        'model.optimizer_state_restored = True',
        'model.densification_state_restored = True',
        'validate_complete_state(',
    ):
        if checkpoint.count(anchor)<1:
            raise RuntimeError(f'Checkpoint overlay lacks required restore contract: {anchor}')
    pseudo=(root/'evidence_track/diffusion/pseudo_manifest.py').read_text(encoding='utf-8')
    for anchor in (
        '"manifest_schema"',
        '"track_h5"',
        '"source_camera_names"',
        'Difix run metadata is required for strict target validation',
    ):
        if anchor not in pseudo:
            raise RuntimeError(f'Pseudo-manifest overlay lacks provenance contract: {anchor}')
    identity=(root/'evidence_track/diffusion/control_identity.py').read_text(encoding='utf-8')
    for anchor in (
        'CONTROLLED_CHECKPOINT_PROVENANCE_SCHEMA = "controlled_checkpoint_provenance_v2"',
        'def canonical_source_image_inventory(',
        'def canonical_training_camera_inventory(',
        'def canonical_controlled_training_protocol(',
        'def canonical_controlled_checkpoint_provenance(',
        'def validate_a0_checkpoint_provenance(',
        '"start_checkpoint_controlled_provenance_sha256"',
        '"pseudo_manifest_sha256"',
        '"pseudo_camera_pool_sha256"',
        'def validate_controlled_checkpoint_provenance_files(',
        '"controlled_training_protocol_sha256"',
        '"source_training_camera_inventory_sha256"',
    ):
        if anchor not in identity:
            raise RuntimeError(
                f'Controlled-identity overlay lacks checkpoint provenance v2 contract: {anchor}'
            )
    checkpoint_provenance=(
        root/'evidence_track/diffusion/checkpoint_state.py'
    ).read_text(encoding='utf-8')
    for anchor in (
        '"controlled_provenance": getattr(model, "controlled_provenance", None)',
        'require_controlled_provenance=role in {"A1", "SELFRENDER", "B"}',
    ):
        if anchor not in checkpoint_provenance:
            raise RuntimeError(
                f'Checkpoint overlay lacks embedded A0 provenance contract: {anchor}'
            )
    result_binding=(
        root/'evidence_track/diffusion/result_binding.py'
    ).read_text(encoding='utf-8')
    for anchor in (
        'RESULT_BINDING_SCHEMA = "controlled_result_binding_v5"',
        '"pseudo_validated_methods"',
        '"final_checkpoints"',
        'def require_final_checkpoint_binding(',
        'checkpoint_path: str | Path',
        'Final {normalized_method} checkpoint bytes differ from the pair audit',
        'checkpoint has the wrong exact A0 lineage',
    ):
        if anchor not in result_binding:
            raise RuntimeError(
                f'Result-binding overlay lacks exact final-checkpoint contract: {anchor}'
            )
    difix=(root/'evidence_track/diffusion/difix_provenance.py').read_text(encoding='utf-8')
    if 'HELDOUT_DIAGNOSTIC_MANIFEST_SCHEMA = "heldout_identity_diagnostic_v2"' not in difix:
        raise RuntimeError('Difix overlay lacks held-out diagnostic schema v2')
    evidence=(root/'evidence_track/diffusion/evidence_protocol.py').read_text(encoding='utf-8')
    for anchor in (
        '"a1_checkpoint_controlled_provenance_sha256"',
        '"b_checkpoint_controlled_provenance_sha256"',
        '"paired_identity_manifest_metadata_v4"',
        'def dinov2_protocol_from_metadata(',
    ):
        if anchor not in evidence:
            raise RuntimeError(f'Evidence overlay lacks paired provenance contract: {anchor}')
    projection=(
        root/'evidence_track/evaluation/check_camera_projection_equivalence.py'
    ).read_text(encoding='utf-8')
    for anchor in (
        'expected_homogeneous_row',
        '"homogeneous_row_max_abs_difference"',
    ):
        if anchor not in projection:
            raise RuntimeError(f'Projection gate lacks affine-matrix contract: {anchor}')
    paired_manifest=(
        root/'evidence_track/evaluation/make_paired_identity_manifest.py'
    ).read_text(encoding='utf-8')
    for anchor in (
        'PAIRED_MANIFEST_SCHEMA = "paired_identity_manifest_v4"',
        'PAIRED_MANIFEST_METADATA_SCHEMA = "paired_identity_manifest_metadata_v4"',
        'require_final_checkpoint_binding(',
        'checkpoint_path=checkpoint_path',
    ):
        if anchor not in paired_manifest:
            raise RuntimeError(f'Paired-manifest overlay lacks v4 contract: {anchor}')
    association=(root/'evidence_track/evaluation/compare_identity_reports.py').read_text(encoding='utf-8')
    if '"schema": "paired_identity_association_v4"' not in association:
        raise RuntimeError('Identity comparison overlay lacks association schema v4')
    geometry=(root/'evidence_track/geometry/repaired_geometry.py').read_text(encoding='utf-8')
    for anchor in (
        '"loss": float(result["loss"].detach().item())',
        '"observation_count": int(result["observation_count"])',
    ):
        if anchor not in geometry:
            raise RuntimeError(f'Geometry overlay lacks smoke-evidence contract: {anchor}')


def main():
    p=argparse.ArgumentParser();p.add_argument('repo',type=Path);a=p.parse_args();r=a.repo
    r=r.resolve()
    if not r.is_dir():raise RuntimeError(f'Integration target is not a directory: {r}')
    marker=r/'.research_revision_v2.json'
    if marker.exists():raise RuntimeError('Revision already applied; use a fresh pinned checkout')
    # The overlay is copied before this integration step.  It owns the final
    # calibrated tools, while this program only rewrites upstream-owned files.
    required_overlay=(
        'evidence_track/diffusion/calibration_guard.py',
        'evidence_track/diffusion/checkpoint_state.py',
        'evidence_track/diffusion/control_identity.py',
        'evidence_track/diffusion/difix_provenance.py',
        'evidence_track/diffusion/evidence_protocol.py',
        'evidence_track/diffusion/pseudo_schedule.py',
        'evidence_track/diffusion/result_binding.py',
        'evidence_track/geometry/repaired_geometry.py',
        'evidence_track/geometry/strict_track_store.py',
        'evidence_track/evaluation/compare_identity_reports.py',
        'evidence_track/evaluation/make_paired_identity_manifest.py',
    )
    missing=[name for name in required_overlay if not (r/name).is_file()]
    if missing:raise RuntimeError(f'Missing source overlay before AST integration: {missing}')
    validate_overlay_contract(r)
    train_source=(r/'train.py').read_text(encoding='utf-8')
    validate_pre_ast_training(train_source)
    edits={}
    path=r/'scene/gaussian_model.py';edits[path]=replace_methods(path.read_text(encoding='utf-8'))
    path=r/'train.py';edits[path]=fix_training(train_source)
    path=r/'gaussian_renderer/__init__.py';text=path.read_text(encoding='utf-8')
    old='confidence = pc.confidence if pipe.use_confidence else torch.ones_like(pc.confidence)'
    if text.count(old)!=1 or 'Confidence count does not match Gaussian count' in text:raise RuntimeError('Unexpected renderer confidence expression')
    text=text.replace(old,'confidence = pc.confidence if pipe.use_confidence else torch.ones_like(pc.get_opacity)\n    if confidence.shape != pc.get_opacity.shape:\n        raise RuntimeError("Confidence count does not match Gaussian count")')
    edits[path]=text
    # Overlay tools already contain one strict calibration call.  Validate that
    # contract rather than rewriting all of them again.
    calibrated_tools=(
        ('check_camera_projection_equivalence.py','calibrate_scene(scene, args.track_h5)', 1),
        ('export_heldout_views.py','calibrate_scene(scene, args.track_path)', 1),
        ('export_pseudo_views.py','calibrate_scene(scene, args.track_path)', 1),
    )
    for name,calibration_call,expected_count in calibrated_tools:
        path=r/'evidence_track'/'evaluation'/name;text=path.read_text(encoding='utf-8')
        if text.count(calibration_call)!=expected_count:
            raise RuntimeError(f'{name} must contain exactly {expected_count} overlay-owned calibration call(s)')
        if 'strict_source_only_geometry' not in text:
            raise RuntimeError(f'{name} lacks its strict source-only gate')
        ast.parse(text)
    path=r/'evidence_track'/'evaluation'/'test_geometry_recovery.py';text=path.read_text(encoding='utf-8')
    if text.count('calibrate_scene(scene, args.track_path)')!=1:
        raise RuntimeError('test_geometry_recovery.py must contain exactly one overlay-owned calibration call')
    if 'strict_source_only_geometry' not in text:
        raise RuntimeError('test_geometry_recovery.py lacks its strict source-only gate')
    recovery_tree=ast.parse(text)
    checkpoint_loads=[
        node for node in ast.walk(recovery_tree)
        if isinstance(node,ast.Call) and ast.unparse(node.func)=='torch.load'
    ]
    if len(checkpoint_loads)!=1:
        raise RuntimeError('Unexpected recovery checkpoint-load structure')
    checkpoint_load=checkpoint_loads[0]
    checkpoint_source=(
        ast.unparse(checkpoint_load.args[0]) if checkpoint_load.args else ''
    )
    weights_only=[
        keyword.value for keyword in checkpoint_load.keywords
        if keyword.arg=='weights_only'
    ]
    if (
        checkpoint_source not in {'args.checkpoint','checkpoint_path'}
        or len(weights_only)!=1
        or not isinstance(weights_only[0],ast.Constant)
        or weights_only[0].value is not False
    ):
        raise RuntimeError('Recovery checkpoint load must explicitly use weights_only=False')
    ast.parse(text)
    for path,text in edits.items():ast.parse(text)
    # Build and parse the full plan before mutation so a source-structure
    # failure occurs before any integration target is changed.
    temporaries=[]
    try:
        for path,text in edits.items():
            temporary=path.with_name(path.name+'.research-v2.tmp')
            with temporary.open('w',encoding='utf-8',newline='\n') as handle:
                handle.write(text)
            temporaries.append((temporary,path))
        for temporary,path in temporaries:os.replace(temporary,path)
        marker_payload=json.dumps({'revision':'research-v2',
                                  'integration_contract':'pre_ast_patch_plus_overlay_then_guarded_ast_v5',
                                  'changed_files':[str(p.relative_to(r)) for p in edits],
                                  'validation':'NOT RUN: assembly transformation only; CPU, CUDA, and end-to-end validation remain pending'},indent=2)+'\n'
        with marker.open('w',encoding='utf-8',newline='\n') as handle:
            handle.write(marker_payload)
    finally:
        for temporary,_ in temporaries:
            if temporary.exists():temporary.unlink()
if __name__=='__main__':main()
