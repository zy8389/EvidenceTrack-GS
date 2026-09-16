from __future__ import annotations
import copy, importlib.util, math, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from diffusion_guidance.calibration_guard import intrinsic_projection_matrix,resized_intrinsics
from diffusion_guidance.camera_utils import camera_fingerprint_from_payload
from diffusion_guidance.evidence_features import SpatialFeatureMap,feature_grid_pixel_coordinates,sample_features
from diffusion_guidance.evidence_matching import error_metrics,select_hard_negatives
from diffusion_guidance.checkpoint_state import capture,restore
from tools.build_anchor_tracks_from_colmap import source_covariance,source_quality
from tools.aggregate_results import paired_bootstrap, filter_by_role
from tools.paired_evidence_report import analyze,VARIANTS
from tools.smoke_test_phase2_1 import camera,observation_for
integration_path = ROOT/'scripts/apply_integration_fixes.py'
if not integration_path.exists(): integration_path = ROOT/'research_scripts/apply_integration_fixes.py'
spec=importlib.util.spec_from_file_location('integration',integration_path)
integration=importlib.util.module_from_spec(spec);spec.loader.exec_module(integration)

@pytest.mark.parametrize('scale',[1e-6,1e-3,10.,1e3,1e6])
def test_covariance_scale(scale):
    cams,ims={},{};x=np.array([.1,-.05,2.5]);obs=[]
    for i,c in enumerate([-.4,0.,.4],1):
        cams[i],ims[i]=camera(c,i);obs.append(observation_for(x,ims[i],cams[i]))
    base=source_covariance(x,obs,cams,ims,1.)
    scaled=copy.deepcopy(ims)
    for im in scaled.values():im['tvec']*=scale
    cov=source_covariance(x*scale,obs,cams,scaled,1.)
    assert np.allclose(cov/scale**2,base,rtol=1e-7,atol=1e-12)
    assert source_quality(.4,3,cov,.4*scale)==pytest.approx(source_quality(.4,3,base,.4),rel=1e-9)

def test_degenerate_covariance_rejected():
    c,im=camera(0,1);x=np.array([.1,0,2.]);o=observation_for(x,im,c)
    with pytest.raises(ValueError):source_covariance(x,[o,o],{1:c},{1:im},1.)

def test_resize_pixel_centers():
    k=dict(fx=100,fy=100,cx=79.5,cy=59.5,width=160,height=120)
    q=resized_intrinsics(k,80,60)
    assert q['cx']==39.5 and q['cy']==29.5

def test_raster_ndc_matches_intrinsics():
    k=dict(fx=511.,fy=495.,cx=307.2,cy=250.4,width=640,height=480)
    p=intrinsic_projection_matrix(k,.01,100.,dtype=torch.float64)
    x=torch.tensor([.17,-.09,2.8,1.],dtype=torch.float64);v=p@x;ndc=v[:2]/v[3]
    pixel=((ndc+1)*torch.tensor([640.,480.])-1)/2
    expected=torch.tensor([k['fx']*x[0]/x[2]+k['cx'],k['fy']*x[1]/x[2]+k['cy']])
    assert torch.allclose(pixel,expected,atol=1e-10)

def test_intrinsics_in_fingerprint():
    p=dict(R=np.eye(3).tolist(),T=[0,0,0],FoVx=1.,FoVy=1.,width=100,height=100,
           intrinsics=dict(fx=50,fy=50,cx=49.5,cy=49.5,width=100,height=100))
    q=copy.deepcopy(p);q['intrinsics']['cx']+=1
    assert camera_fingerprint_from_payload(p)!=camera_fingerprint_from_payload(q)

def test_patch_centers_and_feature_sampling():
    f=torch.eye(4).reshape(4,2,2)
    m=SpatialFeatureMap(f,(28,28),(28,28),(2,2),(14,14),'test','test')
    xy=feature_grid_pixel_coordinates(m).reshape(-1,2)
    assert torch.allclose(xy,torch.tensor([[6.5,6.5],[20.5,6.5],[6.5,20.5],[20.5,20.5]]))
    assert torch.allclose(sample_features(m,xy),torch.eye(4),atol=1e-6)

def test_failure_is_in_pck_denominator():
    m=error_metrics(np.array([[0,0],[np.nan,np.nan]]),np.zeros((2,2)),feature_stride=14)
    assert m['count']==2 and m['successful_count']==1 and m['pck3']==.5 and m['failure_rate']==.5

def test_all_failures_not_perfect_score():
    m=error_metrics(np.full((3,2),np.nan),np.zeros((3,2)),feature_stride=14)
    assert m['pck16']==0 and m['failure_rate']==1

def result_rows():
    scenes=('flower','fortress','horns')
    return [dict(dataset='LLFF',scene=scene,role='confirmatory',seed=seed,method=m,psnr=20+index+(.5 if m=='B' else 0))
            for index,scene in enumerate(scenes,1) for seed in (1,2) for m in ('A1','B')]

def test_paired_bootstrap_cluster_count():
    r=paired_bootstrap(result_rows(),'psnr',repeats=100)
    assert r['mean_delta']==.5 and r['scene_count']==3 and r['paired_run_count']==6
    assert r['scene_bootstrap_ci95']==[.5,.5]

def test_unpaired_results_rejected():
    with pytest.raises(ValueError):paired_bootstrap(result_rows()[1:],'psnr')

def test_confirmatory_filter_requires_explicit_role():
    assert len(filter_by_role(result_rows(), 'confirmatory')) == len(result_rows())
    with pytest.raises(ValueError): filter_by_role([dict(dataset='x')], 'confirmatory')

def test_unmeasured_results_rejected():
    r=result_rows();r[0]['psnr']='NR'
    with pytest.raises(ValueError):paired_bootstrap(r,'psnr')

def test_duplicate_results_rejected():
    r=result_rows()
    with pytest.raises(ValueError):paired_bootstrap(r+r[:1],'psnr')

def paired_rows():
    return [dict(image_name='x',track_id='1',window_radius=32,target_variant=v,
                 image_width=3,image_height=4,projection_error=2,matching_error=(1 if i==0 else float('nan')),
                 feature_stride=1,hard_negative_valid=False,
                 hard_negative_within_radius=False,
                 hard_negative_selection_mode='missing',
                 hard_shuffled_error=2) for i,v in enumerate(VARIANTS)]

def test_paired_failure_diagonal_penalty():
    r=analyze(paired_rows(),32)['scene_metrics']
    assert r['difix_penalized_error']==5 and r['difix_pck16']==0 and r['difix_failure']==1
    assert r['difix_gain_vs_gs']==-4

def test_paired_mismatch_not_dropped():
    with pytest.raises(ValueError):analyze(paired_rows()[:-1],32)

def test_missing_hard_negative_not_success():
    r=analyze(paired_rows(),32)['scene_metrics'];assert r['gs_identity_margin'] is None

def local_hard_negative_rows(correct_error, wrong_error):
    return [dict(image_name='x',track_id='1',window_radius=32,target_variant=v,
                 image_width=3,image_height=4,projection_error=2,
                 matching_error=correct_error,feature_stride=1,
                 hard_negative_valid=True,hard_negative_within_radius=True,
                 hard_negative_selection_mode='local_hard_negative',
                 hard_shuffled_error=wrong_error) for v in VARIANTS]

def test_hard_negative_selection_marks_nonlocal_fallback_invalid():
    queries=torch.tensor([[1.,0.],[.9,.1],[0.,1.]])
    selected=select_hard_negatives(
        queries,np.asarray([[0.,0.],[4.,0.],[100.,0.]]),nearby_radius=8.
    )
    assert selected.selection_mode.tolist()==[
        'local_hard_negative','local_hard_negative','global_fallback'
    ]
    assert selected.within_radius.tolist()==[True,True,False]
    assert selected.projection_distance_px[2] > 8.

def test_identity_margin_penalizes_correct_failure():
    metrics=analyze(local_hard_negative_rows(float('nan'),2.),32)['scene_metrics']
    assert metrics['gs_identity_margin_failure_aware']==-3.
    assert metrics['gs_identity_margin_conditional'] is None
    assert metrics['gs_eligible_count']==1
    assert metrics['gs_hard_negative_valid_count']==1
    assert metrics['gs_correct_failure_count']==1
    assert metrics['gs_wrong_failure_count']==0
    assert metrics['gs_both_success_count']==0

def test_identity_margin_penalizes_wrong_failure():
    metrics=analyze(local_hard_negative_rows(1.,float('nan')),32)['scene_metrics']
    assert metrics['gs_identity_margin_failure_aware']==4.
    assert metrics['gs_identity_margin_conditional'] is None
    assert metrics['gs_correct_failure_count']==0
    assert metrics['gs_wrong_failure_count']==1

class TinyModel:
    def __init__(self):
        self.active_sh_degree=0
        self.max_sh_degree=1
        self._xyz=torch.nn.Parameter(torch.tensor([[.1,.2,.3]]))
        self._features_dc=torch.nn.Parameter(torch.zeros(1,1,3))
        self._features_rest=torch.nn.Parameter(torch.zeros(1,3,3))
        self._scaling=torch.nn.Parameter(torch.zeros(1,3))
        self._rotation=torch.nn.Parameter(torch.tensor([[1.,0.,0.,0.]]))
        self._opacity=torch.nn.Parameter(torch.tensor([[.1]]))
        self.max_radii2D=torch.zeros(1)
        self.confidence=torch.ones(1,1);self.init_point=self._xyz.detach().clone();self.bg_color=torch.empty(0)
        self.spatial_lr_scale=2.;self.training_setup(None)
    @property
    def get_xyz(self):return self._xyz
    @property
    def get_opacity(self):return self._opacity.sigmoid()
    def training_setup(self,opt):
        groups=[
            {'params':[self._xyz],'name':'xyz'},
            {'params':[self._features_dc],'name':'f_dc'},
            {'params':[self._features_rest],'name':'f_rest'},
            {'params':[self._opacity],'name':'opacity'},
            {'params':[self._scaling],'name':'scaling'},
            {'params':[self._rotation],'name':'rotation'},
        ]
        self.optimizer=torch.optim.Adam(groups,lr=.01)
        self.xyz_gradient_accum=torch.zeros(1,1);self.denom=torch.zeros(1,1)

def test_checkpoint_adam_continuation_exact():
    m=TinyModel()
    for _ in range(4):
        m.optimizer.zero_grad();sum(value.square().sum() for value in (
            m._xyz,m._features_dc,m._features_rest,m._opacity,m._scaling,m._rotation
        )).backward();m.optimizer.step()
    state=copy.deepcopy(capture(m));n=TinyModel();restore(n,state,SimpleNamespace(controlled_ab_role='A0'))
    for model in (m,n):
        model.optimizer.zero_grad();sum(value.square().sum() for value in (
            model._xyz,model._features_dc,model._features_rest,model._opacity,model._scaling,model._rotation
        )).backward();model.optimizer.step()
    assert all(torch.equal(getattr(m,name),getattr(n,name)) for name in (
        '_xyz','_features_dc','_features_rest','_opacity','_scaling','_rotation'
    ))
    assert n.optimizer_state_restored

def test_legacy_checkpoint_rejected():
    with pytest.raises(RuntimeError):restore(TinyModel(),tuple(range(12)),SimpleNamespace(controlled_ab_role='B'))

def test_ast_checkpoint_order_and_final_step():
    source='''def training(dataset,opt,pipe,args):
    tb_writer = prepare_output_and_logger(dataset)
    scene = Scene(args, gaussians, shuffle=False)
    for iteration in range(1, opt.iterations+1):
        with torch.no_grad():
            if iteration > first_iter and (iteration in saving_iterations):
                scene.save(iteration)
            if iteration > first_iter and (iteration in checkpoint_iterations):
                torch.save((gaussians.capture(), iteration), 'file')
            if iteration < opt.iterations:
                gaussians.optimizer.step()
            if iteration % 100 == 0:
                gaussians.reset_opacity()
'''
    out=integration.fix_training(source)
    assert out.index('optimizer.step()')<out.index('scene.save(iteration)')
    assert out.index('reset_opacity()')<out.index('gaussians.capture()')
    assert 'if iteration <= opt.iterations:' in out and 'calibrate_scene(scene' in out

def test_ast_method_replacement():
    source='''class GaussianModel:
    def capture(self):
        return 1
    def restore(self, model_args, training_args):
        pass
'''
    out=integration.replace_methods(source)
    assert 'checkpoint_state import capture' in out and 'checkpoint_state import restore' in out
