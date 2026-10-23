"""Keep final report artifacts inside their automatically committed operation."""
import prepare_mlsys_campaign_v001 as original

prior_plan=original.build_plan
def build_plan(*args,**kwargs):
    plan=prior_plan(*args,**kwargs)
    report=next(stage for stage in plan['stages'] if stage['id']=='report')
    report['command'][-1]='{cohort}/operations/report/artifacts'
    return plan

original.build_plan=build_plan
if __name__=='__main__':original.main()
