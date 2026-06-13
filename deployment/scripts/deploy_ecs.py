"""Deploy ARTF containers + orchestrator to ECS Fargate with ALB.

Creates or updates (idempotently):
- VPC with public + private subnets across 2 AZs
- Internet Gateway + NAT Gateway
- ALB in public subnets fronting the orchestrator
- ECS Fargate cluster with 5 services:
  - 4 ARTF containers (internal, service-discovery via CloudMap)
  - 1 orchestrator (behind ALB)
- CloudMap namespace for service discovery between containers

All resources are tagged with the stack name for easy cleanup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time

import boto3
from botocore.exceptions import ClientError

_LOG = logging.getLogger("deploy_ecs")


def _uid(stack_name: str, account_id: str, region: str) -> str:
    """Deterministic 8-char hex suffix unique to this stack+account+region."""
    return hashlib.sha256(f"{stack_name}:{account_id}:{region}".encode()).hexdigest()[:8]

CONTAINERS = [
    {"name": "dlrm-bid-shader", "repo_suffix": "-dlrm-bid-shader", "port": 8081, "health_port": 8080, "env_key": "DLRM_URL"},
    {"name": "widedeep-segment", "repo_suffix": "-widedeep-segment-activator", "port": 8081, "health_port": 8080, "env_key": "WIDEDEEP_URL"},
    {"name": "ncf-deal-manager", "repo_suffix": "-ncf-deal-manager", "port": 8081, "health_port": 8080, "env_key": "NCF_URL"},
    {"name": "metrics-enricher", "repo_suffix": "-metrics-enricher", "port": 8081, "health_port": 8080, "env_key": "METRICS_URL"},
]


def _tag(stack: str) -> list[dict]:
    """Tag format for all AWS APIs (uppercase Key/Value)."""
    return [{"Key": "Project", "Value": stack}, {"Key": "ManagedBy", "Value": "deploy-script"}]


def _ecs_tag(stack: str) -> list[dict]:
    """Tag format for ECS APIs (lowercase key/value)."""
    return [{"key": "Project", "value": stack}, {"key": "ManagedBy", "value": "deploy-script"}]


def _cf_tag(stack: str) -> list[dict]:
    """Alias for _tag — kept for readability in EC2/ELB calls."""
    return _tag(stack)


def deploy(*, stack_name: str, region: str, account_id: str, image_tag: str) -> dict:
    """Deploy the full ECS stack. Returns outputs dict."""
    ec2 = boto3.client("ec2", region_name=region)
    ecs = boto3.client("ecs", region_name=region)
    elbv2 = boto3.client("elbv2", region_name=region)
    logs = boto3.client("logs", region_name=region)
    iam = boto3.client("iam", region_name=region)
    sd = boto3.client("servicediscovery", region_name=region)

    registry = f"{account_id}.dkr.ecr.{region}.amazonaws.com"
    uid = _uid(stack_name, account_id, region)
    outputs = {}

    # -----------------------------------------------------------------
    # VPC
    # -----------------------------------------------------------------
    _LOG.info("Ensuring VPC")
    vpcs = ec2.describe_vpcs(Filters=[{"Name": "tag:Project", "Values": [stack_name]}])["Vpcs"]
    if vpcs:
        vpc_id = vpcs[0]["VpcId"]
    else:
        vpc = ec2.create_vpc(CidrBlock="10.10.0.0/16", TagSpecifications=[{"ResourceType": "vpc", "Tags": _cf_tag(stack_name)}])
        vpc_id = vpc["Vpc"]["VpcId"]
        ec2.modify_vpc_attribute(VpcId=vpc_id, EnableDnsSupport={"Value": True})
        ec2.modify_vpc_attribute(VpcId=vpc_id, EnableDnsHostnames={"Value": True})
    _LOG.info("VPC: %s", vpc_id)

    # Internet Gateway
    igws = ec2.describe_internet_gateways(Filters=[{"Name": "attachment.vpc-id", "Values": [vpc_id]}])["InternetGateways"]
    if igws:
        igw_id = igws[0]["InternetGatewayId"]
    else:
        igw = ec2.create_internet_gateway(TagSpecifications=[{"ResourceType": "internet-gateway", "Tags": _cf_tag(stack_name)}])
        igw_id = igw["InternetGateway"]["InternetGatewayId"]
        ec2.attach_internet_gateway(InternetGatewayId=igw_id, VpcId=vpc_id)

    # Get AZs
    azs_resp = ec2.describe_availability_zones(Filters=[{"Name": "state", "Values": ["available"]}])
    azs = [az["ZoneName"] for az in azs_resp["AvailabilityZones"][:2]]

    # Subnets
    def _ensure_subnet(cidr: str, az: str, name: str, public: bool) -> str:
        subs = ec2.describe_subnets(Filters=[
            {"Name": "vpc-id", "Values": [vpc_id]},
            {"Name": "cidr-block", "Values": [cidr]},
        ])["Subnets"]
        if subs:
            return subs[0]["SubnetId"]
        sub = ec2.create_subnet(VpcId=vpc_id, CidrBlock=cidr, AvailabilityZone=az,
                                TagSpecifications=[{"ResourceType": "subnet", "Tags": _cf_tag(stack_name) + [{"Key": "Name", "Value": name}]}])
        sid = sub["Subnet"]["SubnetId"]
        if public:
            ec2.modify_subnet_attribute(SubnetId=sid, MapPublicIpOnLaunch={"Value": True})
        return sid

    pub_sub_1 = _ensure_subnet("10.10.1.0/24", azs[0], f"{stack_name}-pub-1", True)
    pub_sub_2 = _ensure_subnet("10.10.2.0/24", azs[1], f"{stack_name}-pub-2", True)
    priv_sub_1 = _ensure_subnet("10.10.11.0/24", azs[0], f"{stack_name}-priv-1", False)
    priv_sub_2 = _ensure_subnet("10.10.12.0/24", azs[1], f"{stack_name}-priv-2", False)
    pub_subnets = [pub_sub_1, pub_sub_2]
    priv_subnets = [priv_sub_1, priv_sub_2]

    # Route table for public subnets
    rts = ec2.describe_route_tables(Filters=[{"Name": "vpc-id", "Values": [vpc_id]}, {"Name": "tag:Name", "Values": [f"{stack_name}-pub-rt"]}])["RouteTables"]
    if rts:
        pub_rt = rts[0]["RouteTableId"]
    else:
        rt = ec2.create_route_table(VpcId=vpc_id, TagSpecifications=[{"ResourceType": "route-table", "Tags": _cf_tag(stack_name) + [{"Key": "Name", "Value": f"{stack_name}-pub-rt"}]}])
        pub_rt = rt["RouteTable"]["RouteTableId"]
        ec2.create_route(RouteTableId=pub_rt, DestinationCidrBlock="0.0.0.0/0", GatewayId=igw_id)
    for s in pub_subnets:
        try:
            ec2.associate_route_table(RouteTableId=pub_rt, SubnetId=s)
        except ClientError:
            pass

    # NAT Gateway (for private subnets)
    nats = ec2.describe_nat_gateways(Filters=[{"Name": "vpc-id", "Values": [vpc_id]}, {"Name": "state", "Values": ["available"]}])["NatGateways"]
    if nats:
        nat_id = nats[0]["NatGatewayId"]
    else:
        eip = ec2.allocate_address(Domain="vpc", TagSpecifications=[{"ResourceType": "elastic-ip", "Tags": _cf_tag(stack_name)}])
        nat = ec2.create_nat_gateway(SubnetId=pub_sub_1, AllocationId=eip["AllocationId"],
                                     TagSpecifications=[{"ResourceType": "natgateway", "Tags": _cf_tag(stack_name)}])
        nat_id = nat["NatGateway"]["NatGatewayId"]
        _LOG.info("Waiting for NAT Gateway %s...", nat_id)
        waiter = ec2.get_waiter("nat_gateway_available")
        waiter.wait(NatGatewayIds=[nat_id])

    # Private route table
    priv_rts = ec2.describe_route_tables(Filters=[{"Name": "vpc-id", "Values": [vpc_id]}, {"Name": "tag:Name", "Values": [f"{stack_name}-priv-rt"]}])["RouteTables"]
    if priv_rts:
        priv_rt = priv_rts[0]["RouteTableId"]
    else:
        prt = ec2.create_route_table(VpcId=vpc_id, TagSpecifications=[{"ResourceType": "route-table", "Tags": _cf_tag(stack_name) + [{"Key": "Name", "Value": f"{stack_name}-priv-rt"}]}])
        priv_rt = prt["RouteTable"]["RouteTableId"]
        ec2.create_route(RouteTableId=priv_rt, DestinationCidrBlock="0.0.0.0/0", NatGatewayId=nat_id)
    for s in priv_subnets:
        try:
            ec2.associate_route_table(RouteTableId=priv_rt, SubnetId=s)
        except ClientError:
            pass

    # -----------------------------------------------------------------
    # Security Groups (idempotent — rules are additive, duplicates ignored)
    # -----------------------------------------------------------------
    def _ensure_sg(name: str, desc: str) -> str:
        sgs = ec2.describe_security_groups(Filters=[{"Name": "vpc-id", "Values": [vpc_id]}, {"Name": "group-name", "Values": [name]}])["SecurityGroups"]
        if sgs:
            return sgs[0]["GroupId"]
        sg = ec2.create_security_group(GroupName=name, Description=desc, VpcId=vpc_id,
                                       TagSpecifications=[{"ResourceType": "security-group", "Tags": _cf_tag(stack_name)}])
        return sg["GroupId"]

    alb_sg = _ensure_sg(f"{stack_name}-alb-sg", "ALB - public HTTP")
    svc_sg = _ensure_sg(f"{stack_name}-svc-sg", "ECS services - internal")

    # ALB SG: ensure inbound 80 from anywhere (duplicate is a no-op)
    try:
        ec2.authorize_security_group_ingress(GroupId=alb_sg, IpPermissions=[{"IpProtocol": "tcp", "FromPort": 80, "ToPort": 80, "IpRanges": [{"CidrIp": "0.0.0.0/0"}]}])
    except ClientError:
        pass
    # SVC SG: allow from ALB SG + self (container-to-container) on all service ports
    for port in [8000, 8080, 8081, 50051]:
        try:
            ec2.authorize_security_group_ingress(GroupId=svc_sg, IpPermissions=[
                {"IpProtocol": "tcp", "FromPort": port, "ToPort": port, "UserIdGroupPairs": [{"GroupId": svc_sg}]},
            ])
        except ClientError:
            pass
    # ALB SG -> SVC SG on orchestrator port
    try:
        ec2.authorize_security_group_ingress(GroupId=svc_sg, IpPermissions=[
            {"IpProtocol": "tcp", "FromPort": 8000, "ToPort": 8000, "UserIdGroupPairs": [{"GroupId": alb_sg}]},
        ])
    except ClientError:
        pass

    # -----------------------------------------------------------------
    # ALB (create or update security groups / subnets)
    # -----------------------------------------------------------------
    _LOG.info("Ensuring ALB")
    albs = elbv2.describe_load_balancers()["LoadBalancers"]
    alb = next((a for a in albs if a["LoadBalancerName"] == f"{stack_name}-alb"), None)
    if not alb:
        alb = elbv2.create_load_balancer(
            Name=f"{stack_name}-alb", Subnets=pub_subnets, SecurityGroups=[alb_sg],
            Scheme="internet-facing", Type="application",
            Tags=_cf_tag(stack_name),
        )["LoadBalancers"][0]
    else:
        # Update SGs and subnets to match current config
        elbv2.set_security_groups(LoadBalancerArn=alb["LoadBalancerArn"], SecurityGroups=[alb_sg])
        elbv2.set_subnets(LoadBalancerArn=alb["LoadBalancerArn"], Subnets=pub_subnets)
    alb_arn = alb["LoadBalancerArn"]
    alb_dns = alb["DNSName"]
    outputs["AlbDns"] = alb_dns
    _LOG.info("ALB: %s", alb_dns)

    # Target group (create or update health check)
    tgs = elbv2.describe_target_groups()["TargetGroups"]
    tg = next((t for t in tgs if t["TargetGroupName"] == f"{stack_name}-orch-tg"), None)
    if not tg:
        tg = elbv2.create_target_group(
            Name=f"{stack_name}-orch-tg", Protocol="HTTP", Port=8000,
            VpcId=vpc_id, TargetType="ip",
            HealthCheckPath="/health/ready", HealthCheckIntervalSeconds=10,
            Tags=_cf_tag(stack_name),
        )["TargetGroups"][0]
    else:
        # Update health check settings
        elbv2.modify_target_group(
            TargetGroupArn=tg["TargetGroupArn"],
            HealthCheckPath="/health/ready",
            HealthCheckIntervalSeconds=10,
        )
    tg_arn = tg["TargetGroupArn"]

    # Listener (create or update default action to point to current TG)
    listeners = elbv2.describe_listeners(LoadBalancerArn=alb_arn)["Listeners"]
    if not listeners:
        elbv2.create_listener(LoadBalancerArn=alb_arn, Protocol="HTTP", Port=80,
                              DefaultActions=[{"Type": "forward", "TargetGroupArn": tg_arn}])
    else:
        # Update existing listener to point to the correct target group
        elbv2.modify_listener(
            ListenerArn=listeners[0]["ListenerArn"],
            DefaultActions=[{"Type": "forward", "TargetGroupArn": tg_arn}],
        )

    # -----------------------------------------------------------------
    # CloudMap namespace (service discovery for container-to-container)
    # -----------------------------------------------------------------
    _LOG.info("Ensuring CloudMap namespace")
    ns_name = f"{stack_name}.local"
    namespaces = sd.list_namespaces(Filters=[{"Name": "NAME", "Values": [ns_name], "Condition": "EQ"}])["Namespaces"]
    if namespaces:
        ns_id = namespaces[0]["Id"]
    else:
        ns_resp = sd.create_private_dns_namespace(Name=ns_name, Vpc=vpc_id, Tags=_cf_tag(stack_name))
        ns_id = ns_resp["OperationId"]
        # Wait for namespace creation
        for _ in range(30):
            op = sd.get_operation(OperationId=ns_id)
            if op["Operation"]["Status"] == "SUCCESS":
                ns_id = op["Operation"]["Targets"]["NAMESPACE"]
                break
            time.sleep(2)  # nosemgrep: arbitrary-sleep  — polling for namespace creation
        else:
            # Try to find it
            namespaces = sd.list_namespaces(Filters=[{"Name": "NAME", "Values": [ns_name], "Condition": "EQ"}])["Namespaces"]
            ns_id = namespaces[0]["Id"] if namespaces else ""

    # -----------------------------------------------------------------
    # ECS Cluster
    # -----------------------------------------------------------------
    _LOG.info("Ensuring ECS cluster")
    cluster_name = f"{stack_name}-cluster"
    try:
        ecs.create_cluster(clusterName=cluster_name, tags=_ecs_tag(stack_name))
    except ClientError:
        pass

    # -----------------------------------------------------------------
    # IAM roles
    # -----------------------------------------------------------------
    exec_role_name = f"{stack_name}-ecs-exec-{uid}"
    task_role_name = f"{stack_name}-ecs-task-{uid}"

    def _ensure_role(name: str, service: str, policies: list[str]) -> str:
        arn = f"arn:aws:iam::{account_id}:role/{name}"
        try:
            iam.get_role(RoleName=name)
        except ClientError:
            trust = json.dumps({"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": service}, "Action": "sts:AssumeRole"}]})
            iam.create_role(RoleName=name, AssumeRolePolicyDocument=trust, Tags=_cf_tag(stack_name))
            for p in policies:
                iam.attach_role_policy(RoleName=name, PolicyArn=p)
            time.sleep(5)  # nosemgrep: arbitrary-sleep  — IAM role propagation delay
        return arn

    exec_role_arn = _ensure_role(exec_role_name, "ecs-tasks.amazonaws.com", [
        "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy",
    ])
    task_role_arn = _ensure_role(task_role_name, "ecs-tasks.amazonaws.com", [
        "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess",
    ])

    # -----------------------------------------------------------------
    # Log group
    # -----------------------------------------------------------------
    log_group = f"/ecs/{stack_name}"
    try:
        logs.create_log_group(logGroupName=log_group, tags={"Project": stack_name})
    except ClientError:
        pass

    # -----------------------------------------------------------------
    # Register task definitions and create services
    # -----------------------------------------------------------------
    # First: ARTF containers (internal, with CloudMap service discovery)
    container_urls = {}
    for c in CONTAINERS:
        svc_name = c["name"]
        image = f"{registry}/{stack_name}{c['repo_suffix']}:{image_tag}"
        td_family = f"{stack_name}-{svc_name}"

        _LOG.info("Registering task: %s", td_family)
        ecs.register_task_definition(
            family=td_family,
            networkMode="awsvpc",
            requiresCompatibilities=["FARGATE"],
            cpu="512", memory="1024",
            executionRoleArn=exec_role_arn,
            taskRoleArn=task_role_arn,
            containerDefinitions=[{
                "name": svc_name,
                "image": image,
                "portMappings": [
                    {"containerPort": c["port"], "protocol": "tcp"},
                    {"containerPort": c["health_port"], "protocol": "tcp"},
                    {"containerPort": 50051, "protocol": "tcp"},
                ],
                "logConfiguration": {"logDriver": "awslogs", "options": {
                    "awslogs-group": log_group, "awslogs-region": region, "awslogs-stream-prefix": svc_name,
                }},
                "healthCheck": {"command": ["CMD-SHELL", f"python3 -c \"import urllib.request; urllib.request.urlopen('http://localhost:{c['health_port']}/health/ready')\""],
                                "interval": 10, "timeout": 5, "retries": 3, "startPeriod": 30},
            }],
        )

        # CloudMap service
        cm_services = sd.list_services(Filters=[{"Name": "NAMESPACE_ID", "Values": [ns_id], "Condition": "EQ"}])["Services"]
        cm_svc = next((s for s in cm_services if s["Name"] == svc_name), None)
        if not cm_svc:
            cm_svc = sd.create_service(
                Name=svc_name, NamespaceId=ns_id,
                DnsConfig={"DnsRecords": [{"Type": "A", "TTL": 10}]},
                Tags=_tag(stack_name),
            )["Service"]
        cm_svc_arn = cm_svc["Arn"]

        # ECS service
        try:
            ecs.describe_services(cluster=cluster_name, services=[svc_name])["services"]
            existing = [s for s in ecs.describe_services(cluster=cluster_name, services=[svc_name])["services"] if s["status"] == "ACTIVE"]
        except Exception:
            existing = []

        if existing:
            ecs.update_service(cluster=cluster_name, service=svc_name, taskDefinition=td_family, forceNewDeployment=True)
        else:
            ecs.create_service(
                cluster=cluster_name, serviceName=svc_name, taskDefinition=td_family,
                desiredCount=1, launchType="FARGATE",
                networkConfiguration={"awsvpcConfiguration": {"subnets": priv_subnets, "securityGroups": [svc_sg], "assignPublicIp": "DISABLED"}},
                serviceRegistries=[{"registryArn": cm_svc_arn}],
                tags=_ecs_tag(stack_name),
            )

        container_urls[c["env_key"]] = f"http://{svc_name}.{ns_name}:{c['port']}"

    # Orchestrator (behind ALB)
    orch_image = f"{registry}/{stack_name}-orchestrator:{image_tag}"
    orch_family = f"{stack_name}-orchestrator"
    _LOG.info("Registering task: %s", orch_family)

    orch_env = [{"name": k, "value": v} for k, v in container_urls.items()]
    ecs.register_task_definition(
        family=orch_family,
        networkMode="awsvpc",
        requiresCompatibilities=["FARGATE"],
        cpu="512", memory="1024",
        executionRoleArn=exec_role_arn,
        taskRoleArn=task_role_arn,
        containerDefinitions=[{
            "name": "orchestrator",
            "image": orch_image,
            "portMappings": [{"containerPort": 8000, "protocol": "tcp"}],
            "environment": orch_env,
            "logConfiguration": {"logDriver": "awslogs", "options": {
                "awslogs-group": log_group, "awslogs-region": region, "awslogs-stream-prefix": "orchestrator",
            }},
            "healthCheck": {"command": ["CMD-SHELL", "python3 -c \"import urllib.request; urllib.request.urlopen('http://localhost:8000/health/ready')\""],
                            "interval": 10, "timeout": 5, "retries": 3, "startPeriod": 15},
        }],
    )

    try:
        existing_orch = [s for s in ecs.describe_services(cluster=cluster_name, services=["orchestrator"])["services"] if s["status"] == "ACTIVE"]
    except Exception:
        existing_orch = []

    if existing_orch:
        ecs.update_service(cluster=cluster_name, service="orchestrator", taskDefinition=orch_family, forceNewDeployment=True)
    else:
        ecs.create_service(
            cluster=cluster_name, serviceName="orchestrator", taskDefinition=orch_family,
            desiredCount=1, launchType="FARGATE",
            networkConfiguration={"awsvpcConfiguration": {"subnets": priv_subnets, "securityGroups": [svc_sg], "assignPublicIp": "DISABLED"}},
            loadBalancers=[{"targetGroupArn": tg_arn, "containerName": "orchestrator", "containerPort": 8000}],
            tags=_ecs_tag(stack_name),
        )

    _LOG.info("ECS deployment complete. ALB: %s", alb_dns)

    # Write outputs
    outputs_path = os.path.join(os.path.dirname(__file__), "..", ".ecs-outputs.json")
    with open(outputs_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)

    return outputs


def destroy(*, stack_name: str, region: str) -> None:
    """Delete ECS services and cluster. VPC and ALB are retained."""
    ecs = boto3.client("ecs", region_name=region)
    cluster_name = f"{stack_name}-cluster"
    _LOG.info("Destroying ECS services in %s", cluster_name)
    try:
        services = ecs.list_services(cluster=cluster_name)["serviceArns"]
        for svc_arn in services:
            svc_name = svc_arn.split("/")[-1]
            _LOG.info("  Deleting service %s", svc_name)
            ecs.update_service(cluster=cluster_name, service=svc_name, desiredCount=0)
            ecs.delete_service(cluster=cluster_name, service=svc_name, force=True)
        _LOG.info("  Deleting cluster %s", cluster_name)
        ecs.delete_cluster(cluster=cluster_name)
    except ClientError as exc:
        _LOG.warning("ECS cleanup: %s", exc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", required=True, choices=["deploy", "destroy"])
    parser.add_argument("--stack-name", required=True)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--account-id", default="")
    parser.add_argument("--image-tag", default="latest")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if args.action == "deploy":
        if not args.account_id:
            args.account_id = boto3.client("sts", region_name=args.region).get_caller_identity()["Account"]
        deploy(stack_name=args.stack_name, region=args.region, account_id=args.account_id, image_tag=args.image_tag)
    else:
        destroy(stack_name=args.stack_name, region=args.region)
    return 0


if __name__ == "__main__":
    sys.exit(main())
